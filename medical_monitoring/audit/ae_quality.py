"""
Module F1: AE Data Quality Audit

Comprehensive 10-dimension AE quality assessment based on:
- ICH-E6(R2)
- CTCAE v5.0
- MedDRA v20.1

Dimensions:
1. Naming consistency (MedDRA standardization)
2. Date completeness
3. Outcome/data completeness
4. Severity reasonability (vs CTCAE)
5. SAE screening gaps
6. Drug action vs severity consistency
7. Grade 5/death verification
8. NCS dispute audit
9. Cross-table reference
10. Action plan generation
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..parsers.edc_parser import EDCDataset
from ..utils.helpers import Finding, is_date_incomplete, normalize_subject_id


@dataclass
class NamingVariant:
    """A group of naming variants for the same AE concept."""
    standard_term: str
    meddra_pt: str
    variants: list[str] = field(default_factory=list)
    affected_subjects: list[str] = field(default_factory=list)
    impact: str = ""
    query_suggestion: str = ""


@dataclass
class ActionItem:
    """A prioritized action item from the audit."""
    priority: str  # "urgent", "high", "medium"
    item_id: str
    problem: str
    subjects: list[str]
    action: str
    basis: str
    deadline: str


class AEQualityAuditor:
    """10-dimension AE quality audit engine."""

    def __init__(self, ai_engine=None):
        self.ai_engine = ai_engine
        self._ai_normalizer = None
        self._ai_reasoner = None
        if ai_engine is not None:
            from ..ai.normalizer import MedTermNormalizer
            from ..ai.reasoner import ClinicalReasoner
            self._ai_normalizer = MedTermNormalizer(ai_engine)
            self._ai_reasoner = ClinicalReasoner(ai_engine)
        self.findings: list[Finding] = []
        self.naming_variants: list[NamingVariant] = []
        self.action_items: list[ActionItem] = []

    # Known naming variant patterns for common AEs
    NAMING_RULES = [
        {
            "standard": "发热 (Pyrexia)",
            "variants_pattern": r"发烧|发热|低烧|体温升高|fever|pyrexia",
            "impact": "Multiple naming -> safety signal dilution",
        },
        {
            "standard": "贫血 (Anaemia)",
            "variants_pattern": r"贫血|轻度贫血|中度贫血|重度贫血|anemia|anaemia",
            "impact": "Severity in name -> should be in Grade field",
        },
        {
            "standard": "ALT/AST/BILI (separate coding)",
            "variants_pattern": r"肝功能异常|肝功能|转氨酶|ALT|AST|胆红素|aminotransferase|bilirubin",
            "impact": "Generic liver term -> must split into specific enzymes",
        },
        {
            "standard": "Various hemorrhage terms",
            "variants_pattern": r"脑渗血|脑出血|颅内出血|出血性脑梗|蛛网膜|hemorrh|bleed",
            "impact": "Non-standard terms (脑渗血 not in MedDRA)",
        },
        {
            "standard": "Specific electrolyte terms",
            "variants_pattern": r"电解质紊乱|电解质|低钾|高钾|低钠|高钠|低氯|electrolyte",
            "impact": "Generic electrolyte term -> split to specific ions",
        },
        {
            "standard": "Specific coagulation terms",
            "variants_pattern": r"凝血功能异常|凝血功能障碍|凝血|coagul",
            "impact": "Generic coagulation -> split to PLT/INR/APTT/Fibrinogen",
        },
        {
            "standard": "Stroke progression vs recurrence",
            "variants_pattern": r"卒中早期进展|卒中加重|进展性卒中|脑梗死复发|卒中复发",
            "impact": "Progression vs recurrence are different concepts",
        },
    ]

    def run_full_audit(self, dataset: EDCDataset) -> dict[str, Any]:
        """Run all 10 dimensions of AE quality audit."""
        ae_df = dataset.get_table("AE")
        dd_df = dataset.get_table("DD")

        if ae_df is None:
            return {"error": "AE table not found in dataset"}

        self.findings = []
        self.naming_variants = []
        self.action_items = []

        results = {
            "1_naming": self._audit_naming_consistency(ae_df),
            "2_dates": self._audit_date_completeness(ae_df),
            "3_outcomes": self._audit_outcome_completeness(ae_df),
            "4_severity": self._audit_severity_reasonability(ae_df),
            "5_sae_gaps": self._audit_sae_screening_gaps(ae_df),
            "6_drug_action": self._audit_drug_action_consistency(ae_df),
            "7_death": self._audit_death_consistency(ae_df, dd_df),
            "8_ncs": [],  # Requires CTCAE grading results - handled by F2
            "9_cross_ref": self._generate_cross_references(),
            "10_action_plan": self._generate_action_plan(),
        }

        results["summary"] = {
            "total_ae_records": len(ae_df),
            "total_findings": len(self.findings),
            "urgent_items": len([f for f in self.findings if f.severity == Finding.URGENT]),
            "high_items": len([f for f in self.findings if f.severity == Finding.HIGH]),
            "naming_variant_groups": len(self.naming_variants),
        }

        return results

    def _audit_naming_consistency(self, ae_df: pd.DataFrame) -> list[Finding]:
        """Dimension 1: Detect AE naming variants (regex rules + AI normalization)."""
        findings = []
        all_ae_names = ae_df["field_0"].dropna().astype(str).tolist()

        for rule in self.NAMING_RULES:
            pattern = rule["variants_pattern"]
            matching_names = [
                name for name in all_ae_names
                if re.search(pattern, name, re.IGNORECASE)
            ]

            if not matching_names:
                continue

            unique_variants = list(set(matching_names))
            if len(unique_variants) > 1:
                affected = set()
                for _, row in ae_df.iterrows():
                    name = str(row.get("field_0", ""))
                    if re.search(pattern, name, re.IGNORECASE):
                        affected.add(normalize_subject_id(row.get("subject_id")))

                variant = NamingVariant(
                    standard_term=rule["standard"],
                    meddra_pt=rule["standard"],
                    variants=unique_variants[:10],
                    affected_subjects=sorted(affected)[:20],
                    impact=rule["impact"],
                    query_suggestion=f"Standardize to MedDRA PT: {rule['standard']}",
                )
                self.naming_variants.append(variant)

                findings.append(Finding(
                    category="AE Naming Consistency",
                    subject_id="MULTIPLE",
                    description=(
                        f"{len(unique_variants)} variants for '{rule['standard']}': "
                        f"{', '.join(unique_variants[:5])}"
                    ),
                    severity=Finding.HIGH,
                    source_table="AE",
                    recommendation=f"Standardize per MedDRA PT: {rule['standard']}",
                    regulatory_basis="MedDRA v20.1 coding standards",
                ))

        if self._ai_normalizer:
            unique_names = list(set(all_ae_names))
            try:
                normalized = self._ai_normalizer.normalize_terms(unique_names)
                variant_groups = self._ai_normalizer.build_variant_groups(normalized)
                for pt, group in variant_groups.items():
                    originals = list({n.original for n in group})
                    already_found = any(
                        nv.meddra_pt == pt or pt.lower() in nv.standard_term.lower()
                        for nv in self.naming_variants
                    )
                    if already_found or len(originals) < 2:
                        continue
                    variant = NamingVariant(
                        standard_term=pt,
                        meddra_pt=pt,
                        variants=originals[:10],
                        affected_subjects=[],
                        impact="AI-detected naming variants for same MedDRA PT",
                        query_suggestion=f"Standardize to MedDRA PT: {pt}",
                    )
                    self.naming_variants.append(variant)
                    findings.append(Finding(
                        category="AE Naming Consistency (AI)",
                        subject_id="MULTIPLE",
                        description=(
                            f"AI detected {len(originals)} variants for PT '{pt}': "
                            f"{', '.join(originals[:5])}"
                        ),
                        severity=Finding.HIGH,
                        source_table="AE",
                        recommendation=f"Standardize to MedDRA PT: {pt}",
                        regulatory_basis="MedDRA coding standards (AI-assisted)",
                    ))
            except Exception:
                pass

        self.findings.extend(findings)
        return findings

    def _audit_date_completeness(self, ae_df: pd.DataFrame) -> list[Finding]:
        """Dimension 2: Check for incomplete/unknown dates."""
        findings = []

        for _, row in ae_df.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            ae_name = str(row.get("field_0", "")).strip()
            start_date = str(row.get("field_1", "")).strip()
            end_date = str(row.get("field_4", "")).strip() if row.get("field_4") else ""
            sae_flag = str(row.get("field_5", "")).strip()

            is_sae = "是" in sae_flag or "yes" in sae_flag.lower()

            if is_date_incomplete(start_date) or is_date_incomplete(end_date):
                severity = Finding.URGENT if is_sae else Finding.MEDIUM
                findings.append(Finding(
                    category="AE Date Completeness",
                    subject_id=subj,
                    description=f"AE '{ae_name}' has incomplete date (UK/unknown)",
                    severity=severity,
                    source_table="AE",
                    recommendation="Complete date information",
                    regulatory_basis="ICH E6(R2) - Complete source data",
                ))

        self.findings.extend(findings)
        return findings

    def _audit_outcome_completeness(self, ae_df: pd.DataFrame) -> list[Finding]:
        """Dimension 3: Check for missing outcomes, empty AE names, SAE outcomes."""
        findings = []

        for _, row in ae_df.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            ae_name = str(row.get("field_0", "")).strip()
            outcome = str(row.get("field_4", "")).strip() if row.get("field_4") else ""
            sae_flag = str(row.get("field_5", "")).strip()
            is_sae = "是" in sae_flag or "yes" in sae_flag.lower()

            if not ae_name or ae_name.lower() in ("none", "nan", ""):
                findings.append(Finding(
                    category="AE Data Completeness",
                    subject_id=subj,
                    description="AE name is empty/missing",
                    severity=Finding.URGENT,
                    source_table="AE",
                    recommendation="Urgent: Complete AE name or delete empty row",
                    regulatory_basis="ICH E6(R2) - Data completeness",
                ))

            if "不详" in outcome or not outcome:
                severity = Finding.URGENT if is_sae else Finding.HIGH
                findings.append(Finding(
                    category="AE Outcome Completeness",
                    subject_id=subj,
                    description=f"AE '{ae_name}' outcome unknown/missing (SAE={is_sae})",
                    severity=severity,
                    source_table="AE",
                    recommendation="Follow-up for outcome status",
                    regulatory_basis="ICH E6(R2) - SAE reporting completeness",
                ))

        self.findings.extend(findings)
        return findings

    def _audit_severity_reasonability(self, ae_df: pd.DataFrame) -> list[Finding]:
        """Dimension 4: Check severity against CTCAE definitions."""
        findings = []

        # AEs that cannot be Grade 1 per CTCAE v5.0
        no_grade1_aes = [
            (r"心肌梗死|myocardial infarction|MI", 3, "MI has no G1/G2 in CTCAE"),
            (r"肺栓塞|pulmonary embolism|PE", 3, "PE has no G1/G2 in CTCAE"),
            (r"脑卒中|stroke|cerebrovascular", 3, "Stroke is at minimum G3"),
        ]

        for _, row in ae_df.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            ae_name = str(row.get("field_0", "")).strip()
            severity_str = str(row.get("field_2", "")).strip()

            try:
                grade = int(re.search(r"\d", severity_str).group()) if severity_str else None
            except (AttributeError, ValueError):
                grade = None

            if grade is None:
                continue

            for pattern, min_grade, reason in no_grade1_aes:
                if re.search(pattern, ae_name, re.IGNORECASE) and grade < min_grade:
                    findings.append(Finding(
                        category="AE Severity Reasonability",
                        subject_id=subj,
                        description=(
                            f"AE '{ae_name}' Grade {grade} < CTCAE minimum {min_grade}. "
                            f"Reason: {reason}"
                        ),
                        severity=Finding.URGENT,
                        source_table="AE",
                        recommendation=f"Upgrade to >= Grade {min_grade}; assess SAE",
                        regulatory_basis="CTCAE v5.0 grading definitions",
                    ))

        self.findings.extend(findings)
        return findings

    def _audit_sae_screening_gaps(self, ae_df: pd.DataFrame) -> list[Finding]:
        """Dimension 5: Grade 3+ AEs not marked as SAE."""
        findings = []

        for _, row in ae_df.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            ae_name = str(row.get("field_0", "")).strip()
            severity_str = str(row.get("field_2", "")).strip()
            sae_flag = str(row.get("field_5", "")).strip()

            try:
                grade = int(re.search(r"\d", severity_str).group()) if severity_str else None
            except (AttributeError, ValueError):
                grade = None

            is_sae = "是" in sae_flag or "yes" in sae_flag.lower()

            if grade is not None and grade >= 3 and not is_sae:
                findings.append(Finding(
                    category="SAE Screening Gap",
                    subject_id=subj,
                    description=(
                        f"AE '{ae_name}' Grade {grade} not marked as SAE. "
                        f"G3 (hospitalization/disability) usually meets SAE criteria."
                    ),
                    severity=Finding.HIGH,
                    source_table="AE",
                    recommendation="Evaluate if SAE criteria met (ICH-E2A)",
                    regulatory_basis="ICH-E2A SAE definition",
                ))

        self.findings.extend(findings)
        return findings

    def _audit_drug_action_consistency(self, ae_df: pd.DataFrame) -> list[Finding]:
        """Dimension 6: Drug action should be consistent with severity."""
        findings = []

        for _, row in ae_df.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            ae_name = str(row.get("field_0", "")).strip()
            severity_str = str(row.get("field_2", "")).strip()

            try:
                grade = int(re.search(r"\d", severity_str).group()) if severity_str else None
            except (AttributeError, ValueError):
                grade = None

            # Look for drug action field (varies by EDC)
            drug_action = ""
            for field_name in ["field_6", "field_7", "field_8"]:
                val = str(row.get(field_name, "")).strip()
                if "暂停" in val or "停药" in val or "减量" in val:
                    drug_action = val
                    break

            if grade == 1 and ("暂停" in drug_action or "停药" in drug_action):
                findings.append(Finding(
                    category="Drug Action Consistency",
                    subject_id=subj,
                    description=(
                        f"AE '{ae_name}' Grade 1 but drug suspended/stopped - "
                        f"G1 typically does not require dose modification"
                    ),
                    severity=Finding.MEDIUM,
                    source_table="AE",
                    recommendation="Verify: supports grade upgrade OR was action for another AE",
                    regulatory_basis="CTCAE v5.0 - Grade implications",
                ))

        self.findings.extend(findings)
        return findings

    def _audit_death_consistency(
        self, ae_df: pd.DataFrame, dd_df: pd.DataFrame | None
    ) -> list[Finding]:
        """Dimension 7: Grade 5/death events consistency check."""
        findings = []

        fatal_aes = []
        for _, row in ae_df.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            ae_name = str(row.get("field_0", "")).strip()
            severity_str = str(row.get("field_2", "")).strip()
            outcome = str(row.get("field_4", "")).strip()
            causality = str(row.get("field_3", "")).strip()
            sae_type = str(row.get("field_5", "")).strip()

            is_fatal = "5" in severity_str or "死亡" in outcome or "death" in outcome.lower()
            if is_fatal:
                fatal_aes.append({
                    "subject_id": subj,
                    "ae_name": ae_name,
                    "causality": causality,
                    "sae_type": sae_type,
                })

        # Check each fatal AE for completeness
        for fatal in fatal_aes:
            if not fatal["causality"]:
                findings.append(Finding(
                    category="Death Event Audit",
                    subject_id=fatal["subject_id"],
                    description=f"Fatal AE '{fatal['ae_name']}' missing causality assessment",
                    severity=Finding.URGENT,
                    source_table="AE",
                    recommendation="Complete causality assessment for fatal event",
                    regulatory_basis="ICH E6(R2) - Death reporting",
                ))

            if "不明" in fatal["ae_name"] or "unexplained" in fatal["ae_name"].lower():
                findings.append(Finding(
                    category="Death Event Audit",
                    subject_id=fatal["subject_id"],
                    description=f"Unexplained death: '{fatal['ae_name']}' requires investigation",
                    severity=Finding.URGENT,
                    source_table="AE",
                    recommendation="Provide death investigation report or autopsy results",
                    regulatory_basis="GCP - Death cause determination",
                ))

        self.findings.extend(findings)
        return findings

    def _generate_cross_references(self) -> list[dict]:
        """Dimension 9: Generate cross-reference impact matrix."""
        return [
            {
                "related_audit": "Audit-CTCAE分级",
                "finding": "AE severity < CTCAE calculated grade",
                "ae_quality_impact": "Safety signal underestimation",
                "priority_action": "Reconcile grading discrepancies",
            },
            {
                "related_audit": "Audit-LB↔AE",
                "finding": "CS lab abnormalities without AE records",
                "ae_quality_impact": "AE traceability chain broken",
                "priority_action": "Supplement LB→AE links",
            },
            {
                "related_audit": "Audit-药物相关性",
                "finding": "IB known risk AEs assessed as unrelated",
                "ae_quality_impact": "Causality assessment possibly insufficient",
                "priority_action": "Re-evaluate hemorrhage AE causality",
            },
            {
                "related_audit": "Audit-CM↔AE",
                "finding": "CM start before AE onset",
                "ae_quality_impact": "Temporal logic errors in CM-AE relationship",
                "priority_action": "Verify CM-AE chronology",
            },
        ]

    def _generate_action_plan(self) -> list[ActionItem]:
        """Dimension 10: Generate prioritized action plan from all findings."""
        urgent = [f for f in self.findings if f.severity == Finding.URGENT]
        high = [f for f in self.findings if f.severity == Finding.HIGH]
        medium = [f for f in self.findings if f.severity == Finding.MEDIUM]

        items = []
        for i, f in enumerate(urgent, 1):
            items.append(ActionItem(
                priority="urgent",
                item_id=f"U{i}",
                problem=f.description[:100],
                subjects=[f.subject_id],
                action=f.recommendation,
                basis=f.regulatory_basis,
                deadline="24-48h",
            ))

        for i, f in enumerate(high, 1):
            items.append(ActionItem(
                priority="high",
                item_id=f"H{i}",
                problem=f.description[:100],
                subjects=[f.subject_id],
                action=f.recommendation,
                basis=f.regulatory_basis,
                deadline="This week",
            ))

        for i, f in enumerate(medium[:20], 1):
            items.append(ActionItem(
                priority="medium",
                item_id=f"M{i}",
                problem=f.description[:100],
                subjects=[f.subject_id],
                action=f.recommendation,
                basis=f.regulatory_basis,
                deadline="Two weeks",
            ))

        self.action_items = items
        return items
