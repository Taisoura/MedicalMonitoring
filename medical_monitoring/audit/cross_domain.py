"""
Module B: Cross-Domain Audit Engine

Implements configurable cross-domain consistency checks between EDC tables.
Rules are defined in YAML configuration and executed against parsed EDC data.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..parsers.edc_parser import EDCDataset
from ..utils.helpers import (
    Finding,
    findings_to_dataframe,
    normalize_subject_id,
    parse_date_flexible,
)


class CrossDomainAuditor:
    """Executes cross-domain audit rules against an EDCDataset."""

    def __init__(self, rules_config: str | Path | None = None, ai_engine=None):
        self.rules: dict[str, Any] = {}
        self.ai_engine = ai_engine
        self._ai_matcher = None
        if ai_engine is not None:
            from ..ai.matcher import SemanticMatcher
            self._ai_matcher = SemanticMatcher(ai_engine)
        if rules_config:
            self._load_rules(rules_config)

    def _load_rules(self, path: str | Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        self.rules = config.get("rules", {})

    def run_all(self, dataset: EDCDataset) -> dict[str, list[Finding]]:
        """Run all configured audit rules and return findings by rule."""
        results = {}
        for rule_id, rule_config in self.rules.items():
            handler = getattr(self, f"_audit_{rule_id}", None)
            if handler:
                results[rule_id] = handler(dataset, rule_config)
            else:
                results[rule_id] = []
        return results

    def run_single(self, dataset: EDCDataset, rule_id: str) -> list[Finding]:
        """Run a single audit rule by ID."""
        rule_config = self.rules.get(rule_id)
        if not rule_config:
            raise ValueError(f"Unknown rule: {rule_id}")
        handler = getattr(self, f"_audit_{rule_id}", None)
        if handler:
            return handler(dataset, rule_config)
        return []

    def _audit_cm_ae_temporal(self, dataset: EDCDataset, config: dict) -> list[Finding]:
        """CM start date should be AFTER corresponding AE onset date."""
        findings = []
        cm_df = dataset.get_table("CM")
        ae_df = dataset.get_table("AE")
        if cm_df is None or ae_df is None:
            return findings

        threshold = config.get("threshold_days", 7)

        for _, cm_row in cm_df.iterrows():
            subj = normalize_subject_id(cm_row.get("subject_id"))
            if not subj:
                continue

            cm_start = parse_date_flexible(cm_row.get("field_2"))
            drug_name = str(cm_row.get("field_0", "")).strip()
            indication = str(cm_row.get("field_1", "")).strip()

            if cm_start is None or not drug_name:
                continue

            subj_aes = ae_df[ae_df["subject_id"].apply(normalize_subject_id) == subj]
            for _, ae_row in subj_aes.iterrows():
                ae_name = str(ae_row.get("field_0", "")).strip()
                ae_onset = parse_date_flexible(ae_row.get("field_1"))
                if ae_onset is None:
                    continue

                if self._cm_treats_ae(drug_name, indication, ae_name):
                    diff_days = (ae_onset - cm_start).days
                    if diff_days > 0 and diff_days > threshold:
                        findings.append(Finding(
                            category="CM↔AE Temporal",
                            subject_id=subj,
                            description=(
                                f"CM '{drug_name}' started {diff_days}d BEFORE "
                                f"AE '{ae_name}' onset"
                            ),
                            severity=Finding.HIGH if diff_days > 14 else Finding.MEDIUM,
                            source_table="CM/AE",
                            recommendation=f"Verify: CM start {cm_start.date()} vs AE onset {ae_onset.date()}",
                            regulatory_basis="ICH E6(R2) Data Consistency",
                        ))
        return findings

    def _audit_lb_ae_underreporting(self, dataset: EDCDataset, config: dict) -> list[Finding]:
        """Lab results marked clinically significant must have corresponding AE."""
        findings = []
        ae_df = dataset.get_table("AE")
        if ae_df is None:
            return findings

        cs_values = config.get("clinical_significance_values", ["异常有临床意义"])
        lb_tables = ["LB1", "LB2", "LB3"]

        for lb_name in lb_tables:
            lb_df = dataset.get_table(lb_name)
            if lb_df is None:
                continue

            for _, lb_row in lb_df.iterrows():
                subj = normalize_subject_id(lb_row.get("subject_id"))
                cs_field = str(lb_row.get("field_6", "")).strip()

                if not any(csv in cs_field for csv in cs_values):
                    continue

                test_name = str(lb_row.get("field_0", "")).strip()
                test_code = str(lb_row.get("field_1", "")).strip()
                result = str(lb_row.get("field_2", "")).strip()

                subj_aes = ae_df[ae_df["subject_id"].apply(normalize_subject_id) == subj]
                has_matching = self._find_matching_ae_for_lab(
                    test_name, test_code, subj_aes
                )

                if not has_matching:
                    findings.append(Finding(
                        category="LB↔AE Under-reporting",
                        subject_id=subj,
                        description=(
                            f"{lb_name} '{test_name}' ({test_code}) = {result} "
                            f"marked CS but no corresponding AE"
                        ),
                        severity=Finding.HIGH,
                        source_table=lb_name,
                        recommendation="Report AE or provide NCS justification",
                        regulatory_basis="ICH E6(R2) Section 12.2",
                    ))
        return findings

    def _audit_mh_cm_reasonability(self, dataset: EDCDataset, config: dict) -> list[Finding]:
        """Active comorbidities should have corresponding treatment medications."""
        findings = []
        mh_df = dataset.get_table("MH2")
        cm_df = dataset.get_table("CM")
        if mh_df is None or cm_df is None:
            return findings

        matching_rules = config.get("matching_rules", [])

        for _, mh_row in mh_df.iterrows():
            subj = normalize_subject_id(mh_row.get("subject_id"))
            disease = str(mh_row.get("field_1", "")).strip()
            ongoing = str(mh_row.get("field_3", "")).strip()

            if not disease or not subj:
                continue
            if ongoing and "否" in ongoing:
                continue

            for rule in matching_rules:
                condition_pattern = rule["condition"]
                if not re.search(condition_pattern, disease, re.IGNORECASE):
                    continue

                subj_cms = cm_df[cm_df["subject_id"].apply(normalize_subject_id) == subj]
                drug_pattern = rule["expected_drugs"]
                has_drug = False

                for _, cm_row in subj_cms.iterrows():
                    drug_name = str(cm_row.get("field_0", "")).strip()
                    if re.search(drug_pattern, drug_name, re.IGNORECASE):
                        has_drug = True
                        break

                if not has_drug:
                    findings.append(Finding(
                        category="MH↔CM Reasonability",
                        subject_id=subj,
                        description=(
                            f"Comorbidity '{disease}' without "
                            f"{rule['drug_class']} medication"
                        ),
                        severity=Finding.HIGH,
                        source_table="MH2/CM",
                        recommendation=f"Verify treatment for {disease}",
                        regulatory_basis=rule.get("clinical_risk", "GCP compliance"),
                    ))
        return findings

    def _audit_ae_dd_death_consistency(self, dataset: EDCDataset, config: dict) -> list[Finding]:
        """AE death outcomes must match DD records."""
        findings = []
        ae_df = dataset.get_table("AE")
        dd_df = dataset.get_table("DD")
        if ae_df is None or dd_df is None:
            return findings

        dd_subjects = set()
        for _, dd_row in dd_df.iterrows():
            subj = normalize_subject_id(dd_row.get("subject_id"))
            death_flag = str(dd_row.get("field_0", "")).strip()
            if "是" in death_flag or "yes" in death_flag.lower():
                dd_subjects.add(subj)

        ae_death_subjects = set()
        for _, ae_row in ae_df.iterrows():
            subj = normalize_subject_id(ae_row.get("subject_id"))
            outcome = str(ae_row.get("field_4", "")).strip()
            severity = str(ae_row.get("field_2", "")).strip()
            if "死亡" in outcome or "death" in outcome.lower() or severity == "5":
                ae_death_subjects.add(subj)

        for subj in ae_death_subjects - dd_subjects:
            findings.append(Finding(
                category="AE↔DD Death Consistency",
                subject_id=subj,
                description="AE outcome=death but no DD record found",
                severity=Finding.URGENT,
                source_table="AE/DD",
                recommendation="Urgent: Complete DD form for this subject",
                regulatory_basis="ICH E6(R2) SAE/Death reporting",
            ))

        for subj in dd_subjects - ae_death_subjects:
            findings.append(Finding(
                category="AE↔DD Death Consistency",
                subject_id=subj,
                description="DD records death but no fatal AE (Grade 5) found",
                severity=Finding.URGENT,
                source_table="DD/AE",
                recommendation="Urgent: Record fatal AE for this subject",
                regulatory_basis="ICH E6(R2) SAE/Death reporting",
            ))

        return findings

    def _audit_imaging_ecg_ae_consistency(self, dataset: EDCDataset, config: dict) -> list[Finding]:
        """Clinically significant imaging/ECG findings must have AE."""
        findings = []
        ae_df = dataset.get_table("AE")
        if ae_df is None:
            return findings

        cs_values = ["异常有临床意义", "Abnormal - Clinically Significant"]

        for table_name in ["EG", "MO", "MO1"]:
            df = dataset.get_table(table_name)
            if df is None:
                continue

            for _, row in df.iterrows():
                subj = normalize_subject_id(row.get("subject_id"))
                for col in row.index:
                    val = str(row.get(col, "")).strip()
                    if any(csv in val for csv in cs_values):
                        subj_aes = ae_df[
                            ae_df["subject_id"].apply(normalize_subject_id) == subj
                        ]
                        if subj_aes.empty:
                            findings.append(Finding(
                                category="Imaging/ECG↔AE",
                                subject_id=subj,
                                description=(
                                    f"{table_name} has CS abnormality but no AE for subject"
                                ),
                                severity=Finding.HIGH,
                                source_table=table_name,
                                recommendation="Verify if AE should be reported",
                                regulatory_basis="ICH E6(R2) - CS findings reporting",
                            ))
                        break
        return findings

    def _audit_dynamic_link_accuracy(self, dataset: EDCDataset, config: dict) -> list[Finding]:
        """Verify that EDC dynamic links from LB to AE are medically logical."""
        findings = []
        lb_df = dataset.get_table("LB1")
        ae_df = dataset.get_table("AE")
        if lb_df is None or ae_df is None:
            return findings

        for _, lb_row in lb_df.iterrows():
            subj = normalize_subject_id(lb_row.get("subject_id"))
            link_field = str(lb_row.get("field_7", "")).strip()
            test_name = str(lb_row.get("field_0", "")).strip()

            if not link_field or link_field.lower() in ("", "none", "nan"):
                continue

            if not self._is_link_medically_logical(test_name, link_field):
                findings.append(Finding(
                    category="Dynamic Link Accuracy",
                    subject_id=subj,
                    description=(
                        f"LB '{test_name}' linked to AE '{link_field}' - "
                        f"medical logic mismatch"
                    ),
                    severity=Finding.MEDIUM,
                    source_table="LB1",
                    recommendation="Verify link target matches lab finding",
                    regulatory_basis="Data integrity",
                ))
        return findings

    # --- Helper methods ---

    def _cm_treats_ae(self, drug: str, indication: str, ae_name: str) -> bool:
        """Check if a CM drug is likely treating the given AE (rule + AI fallback)."""
        if not indication and not ae_name:
            return False
        combined = f"{indication} {ae_name}".lower()
        drug_lower = drug.lower()
        treat_patterns = [
            (r"发热|fever", r"布洛芬|对乙酰|退热|ibuprofen|acetaminophen"),
            (r"感染|infection", r"抗生素|头孢|莫西|左氧|antibiotic|cef"),
            (r"疼痛|pain", r"止痛|镇痛|analgesic|tramadol"),
            (r"出血|hemorrh", r"止血|tranexamic|氨甲环酸"),
        ]
        for ae_pat, drug_pat in treat_patterns:
            if re.search(ae_pat, combined) and re.search(drug_pat, drug_lower):
                return True
        if indication and ae_name and ae_name.lower() in indication.lower():
            return True

        if self._ai_matcher:
            result = self._ai_matcher.cm_treats_ae(drug, indication, ae_name)
            if result.confidence >= 0.7:
                return result.matches

        return False

    def _find_matching_ae_for_lab(
        self, test_name: str, test_code: str, ae_df: pd.DataFrame
    ) -> bool:
        """Check if there's a plausible AE matching a lab abnormality (rule + AI fallback)."""
        if ae_df.empty:
            return False
        lab_ae_map = {
            "WBC": ["白细胞", "leukocyte", "wbc"],
            "NEUT": ["中性粒细胞", "neutrop"],
            "LYM": ["淋巴细胞", "lymphocyte"],
            "HGB": ["贫血", "anemia", "血红蛋白"],
            "PLT": ["血小板", "platelet", "thrombocyto"],
            "ALT": ["alt", "转氨酶", "肝功能", "aminotransferase"],
            "AST": ["ast", "转氨酶", "肝功能", "aminotransferase"],
            "CREAT": ["肌酐", "肾功能", "creatinine", "renal"],
            "K": ["低钾", "高钾", "hypokalemia", "hyperkalemia", "钾"],
        }
        code_upper = test_code.upper() if test_code else ""
        keywords = lab_ae_map.get(code_upper, [test_name.lower()])

        for _, ae_row in ae_df.iterrows():
            ae_name = str(ae_row.get("field_0", "")).lower()
            if any(kw.lower() in ae_name for kw in keywords):
                return True

        if self._ai_matcher:
            for _, ae_row in ae_df.iterrows():
                ae_name = str(ae_row.get("field_0", "")).strip()
                if not ae_name:
                    continue
                result = self._ai_matcher.lab_matches_ae(
                    test_name=test_name, test_code=test_code,
                    result_value="", ref_range="", ae_name=ae_name,
                )
                if result.matches and result.confidence >= 0.7:
                    return True

        return False

    def _is_link_medically_logical(self, test_name: str, linked_ae: str) -> bool:
        """Check if a dynamic link from LB test to AE is medically reasonable."""
        test_lower = test_name.lower()
        ae_lower = linked_ae.lower()
        logical_pairs = {
            "plt": ["血小板", "出血", "hemorrh", "bleed"],
            "hgb": ["贫血", "anemia", "出血"],
            "alt": ["肝", "liver", "转氨酶", "hepat"],
            "ast": ["肝", "liver", "转氨酶", "心肌", "hepat"],
            "wbc": ["感染", "infect", "白细胞", "leuko"],
            "lym": ["淋巴", "lymph", "免疫", "感染"],
            "fibrino": ["凝血", "coagul", "出血", "fibrin"],
        }
        for key, valid_terms in logical_pairs.items():
            if key in test_lower:
                return any(t in ae_lower for t in valid_terms)
        return True  # default: assume ok if no specific rule
