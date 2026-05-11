"""
Module E: Drug Causality Audit

Implements bidirectional causality assessment:
- Forward: IB known risks that should be 'related' but assessed as 'unrelated'
- Reverse: AEs assessed as 'related' that may have stronger confounding factors
- Quantitative confounding factor scoring framework

Supports three information-level modes:
- FULL: Complete IB + Protocol provided (drug name, risk mappings, pharmacology)
- PARTIAL: Only drug name or drug class known (no IB risk details)
- BLIND: No IB/Protocol information (pure data-driven + AI inference)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd

from ..parsers.edc_parser import EDCDataset
from ..utils.helpers import Finding, normalize_subject_id


class InfoLevel(Enum):
    """Information availability level for IB/Protocol context."""
    FULL = "full"        # IB + Protocol fully provided
    PARTIAL = "partial"  # Drug name / class known, no detailed IB risk mappings
    BLIND = "blind"      # No IB/Protocol information at all


@dataclass
class DrugContext:
    """
    Encapsulates all available IB/Protocol information about the study drug.

    Users may provide anything from nothing (confidential trial) to full IB
    risk profiles. The framework adapts its analysis depth accordingly.

    Examples:
        # Full IB context
        ctx = DrugContext(
            drug_name="Y-6 Sublingual Tablet",
            drug_class="antiplatelet (PDE3 inhibitor)",
            indication="acute ischemic stroke",
            ib_version="IB v7.0, 2026-01",
            mechanism_summary="Cilostazol PDE3 inhibition → vasodilation + antiplatelet",
            known_risk_summary="Bleeding, headache, hepatotoxicity, tachycardia",
        )

        # Partial: only drug name
        ctx = DrugContext(drug_name="Drug X")

        # Blind: nothing
        ctx = DrugContext()
    """
    drug_name: str = ""
    drug_class: str = ""
    indication: str = ""
    ib_version: str = ""
    mechanism_summary: str = ""
    known_risk_summary: str = ""
    protocol_title: str = ""
    protocol_phase: str = ""
    sponsor: str = ""
    therapeutic_area: str = ""

    @property
    def info_level(self) -> InfoLevel:
        """Auto-detect the information availability level."""
        has_name = bool(self.drug_name.strip())
        has_risks = bool(self.known_risk_summary.strip()) or bool(self.mechanism_summary.strip())
        if has_name and has_risks:
            return InfoLevel.FULL
        if has_name:
            return InfoLevel.PARTIAL
        return InfoLevel.BLIND

    def to_prompt_context(self) -> str:
        """Format available information into a prompt-friendly string."""
        lines = []
        if self.drug_name:
            lines.append(f"研究药物: {self.drug_name}")
        if self.drug_class:
            lines.append(f"药物类别: {self.drug_class}")
        if self.indication:
            lines.append(f"适应症: {self.indication}")
        if self.mechanism_summary:
            lines.append(f"作用机制: {self.mechanism_summary}")
        if self.known_risk_summary:
            lines.append(f"IB已知风险: {self.known_risk_summary}")
        if self.protocol_phase:
            lines.append(f"临床阶段: {self.protocol_phase}")
        if self.therapeutic_area:
            lines.append(f"治疗领域: {self.therapeutic_area}")
        if not lines:
            return "（未提供研究药物/方案信息 -- 涉密/敏感性考虑）"
        return "\n".join(lines)

    def to_dict(self) -> dict[str, str]:
        """Serialize non-empty fields for reporting."""
        return {k: v for k, v in {
            "drug_name": self.drug_name,
            "drug_class": self.drug_class,
            "indication": self.indication,
            "ib_version": self.ib_version,
            "mechanism_summary": self.mechanism_summary,
            "known_risk_summary": self.known_risk_summary,
            "protocol_title": self.protocol_title,
            "protocol_phase": self.protocol_phase,
            "info_level": self.info_level.value,
        }.items() if v}


@dataclass
class IBRiskMapping:
    """Mapping from IB known risks to expected AE types."""
    risk_category: str
    ib_section: str
    expected_ae_patterns: list[str]
    description: str


@dataclass
class ConfoundingScore:
    """Score for a single confounding factor."""
    factor_name: str
    dimension: str  # "concomitant_drug", "baseline", "laboratory"
    score: int
    evidence: str


@dataclass
class CausalityConfig:
    """
    Configuration for drug causality audit.

    Supports three usage modes based on available information:

    1. FULL mode (IB provided):
       Supply drug_name + ib_risk_mappings + cm_drug_scores etc.
       → Forward audit matches IB risks against AE assessments.

    2. PARTIAL mode (drug name only):
       Supply drug_name, optionally drug_context with drug_class/indication.
       → Forward audit skipped; reverse audit + consistency audit run.
       → AI (if enabled) infers plausible risk categories from drug class.

    3. BLIND mode (no IB/Protocol):
       Leave drug_name empty.
       → Forward audit skipped; reverse audit uses generic scoring.
       → Consistency audit uses AE-name clustering (no IB categories).
       → AI (if enabled) provides purely data-driven observations.
    """

    drug_name: str = ""
    drug_context: DrugContext = field(default_factory=DrugContext)
    ib_risk_mappings: list[IBRiskMapping] = field(default_factory=list)

    cm_drug_scores: dict[str, int] = field(default_factory=lambda: {
        "tirofiban": 5,
        "heparin": 4,
        "enoxaparin": 4,
        "aspirin": 3,
        "clopidogrel": 3,
        "warfarin": 4,
        "rivaroxaban": 4,
    })

    baseline_scores: dict[str, int] = field(default_factory=lambda: {
        "coagulation_abnormal": 4,
        "ulcer_history": 3,
        "ckd": 2,
        "liver_disease": 3,
        "thrombocytopenia": 3,
    })

    lab_scores: dict[str, int] = field(default_factory=lambda: {
        "inr_elevated": 3,
        "aptt_elevated": 3,
        "fibrinogen_low": 3,
        "platelet_low": 3,
    })

    drug_fixed_score: int = 1

    strong_downgrade_threshold: int = 8
    suggest_downgrade_threshold: int = 5
    case_by_case_threshold: int = 3

    @property
    def info_level(self) -> InfoLevel:
        """Determine the IB/Protocol information level for this config."""
        if self.ib_risk_mappings:
            return InfoLevel.FULL
        if self.drug_context.info_level != InfoLevel.BLIND:
            return self.drug_context.info_level
        if self.drug_name:
            return InfoLevel.PARTIAL
        return InfoLevel.BLIND


# Pre-built config for Y-6 (D-Borneol + Cilostazol) -- FULL mode example
Y6_CAUSALITY_CONFIG = CausalityConfig(
    drug_name="Y-6 Sublingual Tablet (D-Borneol 6mg + Cilostazol 25mg)",
    drug_context=DrugContext(
        drug_name="Y-6 Sublingual Tablet (D-Borneol 6mg + Cilostazol 25mg)",
        drug_class="antiplatelet (PDE3 inhibitor) + neuroprotectant",
        indication="acute ischemic stroke (within 48h onset)",
        mechanism_summary="Cilostazol: PDE3 inhibition → antiplatelet + vasodilation; D-Borneol: BBB permeability + neuroprotection",
        known_risk_summary="Bleeding (antiplatelet mechanism), headache (vasodilation), hepatotoxicity (D-Borneol metabolism), tachycardia (PDE3)",
        protocol_phase="Phase II/III",
        therapeutic_area="neurology / cerebrovascular",
    ),
    ib_risk_mappings=[
        IBRiskMapping(
            risk_category="hemorrhage",
            ib_section="IB 6.2",
            expected_ae_patterns=[
                "出血", "hemorrh", "ICH", "颅内出血", "消化道出血",
                "血尿", "皮下出血", "蛛网膜下腔", "脑出血",
            ],
            description="Bleeding risk from antiplatelet mechanism (cilostazol)",
        ),
        IBRiskMapping(
            risk_category="headache",
            ib_section="IB 6.1",
            expected_ae_patterns=["头痛", "头晕", "headache", "dizziness"],
            description="Headache from vasodilation (cilostazol PDE3 inhibition)",
        ),
        IBRiskMapping(
            risk_category="hepatotoxicity",
            ib_section="IB 6.3",
            expected_ae_patterns=[
                "肝", "ALT", "AST", "转氨酶", "liver", "hepat",
                "胆红素", "bilirubin",
            ],
            description="Hepatotoxicity (D-Borneol hepatic metabolism)",
        ),
        IBRiskMapping(
            risk_category="cardiac",
            ib_section="IB 6.4",
            expected_ae_patterns=[
                "心动过速", "tachycardia", "心悸", "palpitation",
                "心律", "arrhyth",
            ],
            description="Cardiac effects from PDE3 inhibition",
        ),
    ],
    drug_fixed_score=1,
)


class CausalityAuditor:
    """Bidirectional drug causality audit engine."""

    def __init__(self, config: CausalityConfig | None = None, ai_engine=None):
        self.config = config or CausalityConfig()
        self.ai_engine = ai_engine
        self._ai_reasoner = None
        self._ai_matcher = None
        if ai_engine is not None:
            from ..ai.reasoner import ClinicalReasoner
            from ..ai.matcher import SemanticMatcher
            self._ai_reasoner = ClinicalReasoner(ai_engine)
            self._ai_matcher = SemanticMatcher(ai_engine)

    def run_full_audit(self, dataset: EDCDataset) -> dict[str, Any]:
        """
        Run complete causality audit, adapting to IB information availability.

        - FULL: forward + reverse + consistency (all IB-driven)
        - PARTIAL: reverse + consistency (data-driven, AI-assisted if available)
        - BLIND: reverse only (generic confounding scoring, AI-assisted if available)
        """
        ae_df = dataset.get_table("AE")
        cm_df = dataset.get_table("CM")
        mh_df = dataset.get_table("MH2")
        lb_df = dataset.get_table("LB3")

        level = self.config.info_level
        results: dict[str, Any] = {
            "info_level": level.value,
            "forward_findings": [],
            "reverse_findings": [],
            "consistency_findings": [],
            "ai_causality_assessments": [],
            "summary": {},
        }

        if level == InfoLevel.FULL:
            results["forward_findings"] = self.forward_audit(ae_df)

        results["reverse_findings"] = self.reverse_audit(ae_df, cm_df, mh_df, lb_df)

        if level in (InfoLevel.FULL, InfoLevel.PARTIAL):
            results["consistency_findings"] = self.consistency_audit(ae_df)
        else:
            results["consistency_findings"] = self._blind_consistency_audit(ae_df)

        if self._ai_reasoner and ae_df is not None:
            results["ai_causality_assessments"] = self._ai_enhanced_audit(ae_df, cm_df, mh_df)

        fwd = results["forward_findings"]
        rev = results["reverse_findings"]
        results["summary"] = {
            "total_ae": len(ae_df) if ae_df is not None else 0,
            "info_level": level.value,
            "forward_issues": len(fwd),
            "forward_available": level == InfoLevel.FULL,
            "reverse_downgrades": len([r for r in rev if r.severity == Finding.HIGH]),
            "drug_name": self.config.drug_name or "(not provided)",
            "drug_context": self.config.drug_context.to_dict() if self.config.drug_context else {},
        }

        if level != InfoLevel.FULL:
            results["summary"]["info_level_note"] = {
                InfoLevel.PARTIAL.value: (
                    "IB 风险映射未提供 -- 正向审计（IB→AE比对）已跳过。"
                    "建议提供 IB 已知风险以启用完整因果分析。"
                ),
                InfoLevel.BLIND.value: (
                    "研究药物/方案信息未提供 -- 仅执行数据驱动的混杂因素分析。"
                    "正向审计和 IB 一致性审计已跳过。"
                    "如有涉密顾虑，可仅提供药物类别（如'抗血小板药'）而非具体药名。"
                ),
            }.get(level.value, "")

        return results

    def forward_audit(self, ae_df: pd.DataFrame | None) -> list[Finding]:
        """
        Forward audit: Find AEs matching IB known risks
        that are assessed as 'unrelated' or 'possibly unrelated'.

        Requires InfoLevel.FULL (ib_risk_mappings must be non-empty).
        Returns empty list if no IB risk mappings are configured.
        """
        if ae_df is None or not self.config.ib_risk_mappings:
            return []

        findings = []
        unrelated_terms = ["无关", "可能无关", "unrelated", "unlikely"]

        for _, row in ae_df.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            ae_name = str(row.get("field_0", "")).strip()
            causality = str(row.get("field_3", "")).strip()

            if not ae_name or not causality:
                continue

            is_unrelated = any(t in causality for t in unrelated_terms)
            if not is_unrelated:
                continue

            for mapping in self.config.ib_risk_mappings:
                if any(re.search(pat, ae_name, re.IGNORECASE)
                       for pat in mapping.expected_ae_patterns):
                    findings.append(Finding(
                        category=f"Forward Causality ({mapping.risk_category})",
                        subject_id=subj,
                        description=(
                            f"AE '{ae_name}' matches IB risk '{mapping.ib_section}: "
                            f"{mapping.description}' but assessed as '{causality}'"
                        ),
                        severity=Finding.HIGH,
                        source_table="AE",
                        recommendation=(
                            f"Re-evaluate: {mapping.risk_category} is IB known risk"
                        ),
                        regulatory_basis=f"IB Section {mapping.ib_section}",
                    ))
                    break

        return findings

    def reverse_audit(
        self,
        ae_df: pd.DataFrame | None,
        cm_df: pd.DataFrame | None,
        mh_df: pd.DataFrame | None,
        lb_df: pd.DataFrame | None,
    ) -> list[Finding]:
        """
        Reverse audit: For AEs assessed as 'related', check if confounding
        factors (concomitant drugs, baseline conditions, lab values) are stronger.
        """
        if ae_df is None:
            return []

        findings = []
        related_terms = ["有关", "可能有关", "related", "possible"]

        for _, row in ae_df.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            ae_name = str(row.get("field_0", "")).strip()
            causality = str(row.get("field_3", "")).strip()

            is_related = any(t in causality for t in related_terms)
            if not is_related:
                continue

            score = self._compute_confounding_score(subj, ae_name, cm_df, mh_df, lb_df)
            total_confounding = sum(s.score for s in score)
            drug_score = self.config.drug_fixed_score

            if total_confounding >= self.config.strong_downgrade_threshold:
                severity = Finding.HIGH
                action = "Strong downgrade recommendation"
            elif total_confounding >= self.config.suggest_downgrade_threshold:
                severity = Finding.MEDIUM
                action = "Suggest downgrade"
            elif total_confounding >= self.config.case_by_case_threshold:
                severity = Finding.LOW
                action = "Case-by-case review"
            else:
                continue

            factors_str = "; ".join(
                f"{s.factor_name}(+{s.score})" for s in score if s.score > 0
            )

            findings.append(Finding(
                category="Reverse Causality",
                subject_id=subj,
                description=(
                    f"AE '{ae_name}' assessed '{causality}' but confounding "
                    f"score={total_confounding} >> drug score={drug_score}. "
                    f"Factors: {factors_str}"
                ),
                severity=severity,
                source_table="AE/CM/MH2/LB3",
                recommendation=action,
                regulatory_basis="Confounding factor quantitative scoring",
            ))

        return findings

    def consistency_audit(self, ae_df: pd.DataFrame | None) -> list[Finding]:
        """
        Check causality assessment consistency across similar AEs.

        In FULL mode: uses IB risk mappings to group AEs.
        In PARTIAL mode: uses IB mappings if available, falls back to name grouping.
        """
        if ae_df is None:
            return []

        if not self.config.ib_risk_mappings:
            return self._blind_consistency_audit(ae_df)

        findings = []
        for mapping in self.config.ib_risk_mappings:
            matching_aes = []
            for _, row in ae_df.iterrows():
                ae_name = str(row.get("field_0", "")).strip()
                if any(re.search(pat, ae_name, re.IGNORECASE)
                       for pat in mapping.expected_ae_patterns):
                    matching_aes.append({
                        "subject_id": normalize_subject_id(row.get("subject_id")),
                        "ae_name": ae_name,
                        "causality": str(row.get("field_3", "")).strip(),
                    })

            if len(matching_aes) < 2:
                continue

            related = [a for a in matching_aes if "有关" in a["causality"]]
            unrelated = [a for a in matching_aes if "无关" in a["causality"]]

            if related and unrelated:
                findings.append(Finding(
                    category="Causality Consistency",
                    subject_id="MULTIPLE",
                    description=(
                        f"IB risk '{mapping.risk_category}': {len(related)} AEs "
                        f"assessed 'related' but {len(unrelated)} similar AEs "
                        f"assessed 'unrelated'"
                    ),
                    severity=Finding.HIGH,
                    source_table="AE",
                    recommendation="Standardize causality assessment for same AE type",
                    regulatory_basis=f"IB {mapping.ib_section} consistency",
                ))
        return findings

    def _blind_consistency_audit(self, ae_df: pd.DataFrame | None) -> list[Finding]:
        """
        Consistency audit without IB: group AEs by exact name and check
        whether the same AE term receives inconsistent causality assessments
        across different subjects.
        """
        if ae_df is None:
            return []

        from collections import defaultdict
        ae_groups: dict[str, list[dict[str, str]]] = defaultdict(list)

        for _, row in ae_df.iterrows():
            ae_name = str(row.get("field_0", "")).strip().lower()
            causality = str(row.get("field_3", "")).strip()
            if ae_name and causality:
                ae_groups[ae_name].append({
                    "subject_id": normalize_subject_id(row.get("subject_id")),
                    "causality": causality,
                    "original_name": str(row.get("field_0", "")).strip(),
                })

        findings = []
        for ae_name, entries in ae_groups.items():
            if len(entries) < 2:
                continue

            related = [e for e in entries if any(t in e["causality"] for t in ["有关", "related", "possible"])]
            unrelated = [e for e in entries if any(t in e["causality"] for t in ["无关", "unrelated", "unlikely"])]

            if related and unrelated:
                display_name = entries[0]["original_name"]
                findings.append(Finding(
                    category="Causality Consistency (Data-Driven)",
                    subject_id="MULTIPLE",
                    description=(
                        f"AE '{display_name}': {len(related)} instances assessed "
                        f"'related' but {len(unrelated)} assessed 'unrelated' "
                        f"across different subjects"
                    ),
                    severity=Finding.MEDIUM,
                    source_table="AE",
                    recommendation="Review causality assessment consistency for same AE type",
                    regulatory_basis="ICH E6(R2) - Consistency of safety assessments",
                ))

        return findings

    def _ai_enhanced_audit(
        self,
        ae_df: pd.DataFrame,
        cm_df: pd.DataFrame | None,
        mh_df: pd.DataFrame | None,
    ) -> list[dict[str, Any]]:
        """
        AI-assisted causality review for cases where:
        - IB is not available (BLIND/PARTIAL mode) -- AI provides drug class inference
        - AE-drug relationship needs deeper clinical reasoning
        """
        if self._ai_reasoner is None:
            return []

        assessments = []
        drug_ctx = self.config.drug_context
        drug_name = self.config.drug_name or drug_ctx.drug_name or "（未提供）"

        sample_aes = ae_df.head(min(20, len(ae_df)))

        for _, row in sample_aes.iterrows():
            ae_name = str(row.get("field_0", "")).strip()
            causality = str(row.get("field_3", "")).strip()
            severity = str(row.get("field_2", "")).strip()
            subj = normalize_subject_id(row.get("subject_id"))

            if not ae_name:
                continue

            mh_text = ""
            if mh_df is not None:
                subj_mh = mh_df[mh_df["subject_id"].apply(normalize_subject_id) == subj]
                mh_terms = subj_mh["field_1"].dropna().astype(str).tolist() if "field_1" in subj_mh.columns else []
                mh_text = "; ".join(mh_terms[:5])

            cm_text = ""
            if cm_df is not None:
                subj_cm = cm_df[cm_df["subject_id"].apply(normalize_subject_id) == subj]
                cm_names = subj_cm["field_0"].dropna().astype(str).tolist() if "field_0" in subj_cm.columns else []
                cm_text = "; ".join(cm_names[:5])

            try:
                result = self._ai_reasoner.assess_causality(
                    ae_name=ae_name,
                    drug_name=drug_name,
                    indication=drug_ctx.indication,
                    known_risks=drug_ctx.known_risk_summary,
                    severity=severity,
                    investigator_causality=causality,
                    medical_history=mh_text,
                    concomitant_meds=cm_text,
                    subject_id=subj,
                )
                assessments.append({
                    "subject_id": subj,
                    "ae_name": ae_name,
                    "investigator": causality,
                    "ai_suggested": result.suggested_category,
                    "ai_confidence": result.confidence,
                    "ai_reasoning": result.reasoning,
                    "source": "AI-suggested",
                })
            except Exception:
                pass

        return assessments

    def _compute_confounding_score(
        self,
        subject_id: str,
        ae_name: str,
        cm_df: pd.DataFrame | None,
        mh_df: pd.DataFrame | None,
        lb_df: pd.DataFrame | None,
    ) -> list[ConfoundingScore]:
        """Compute multi-dimensional confounding factor scores for a subject."""
        scores = []

        if cm_df is not None:
            subj_cms = cm_df[cm_df["subject_id"].apply(normalize_subject_id) == subject_id]
            for _, cm_row in subj_cms.iterrows():
                drug = str(cm_row.get("field_0", "")).lower()
                for drug_key, points in self.config.cm_drug_scores.items():
                    if drug_key in drug:
                        scores.append(ConfoundingScore(
                            factor_name=drug_key,
                            dimension="concomitant_drug",
                            score=points,
                            evidence=f"CM: {drug}",
                        ))
                        break

        if mh_df is not None:
            subj_mh = mh_df[mh_df["subject_id"].apply(normalize_subject_id) == subject_id]
            for _, mh_row in subj_mh.iterrows():
                disease = str(mh_row.get("field_1", "")).lower()
                for cond_key, points in self.config.baseline_scores.items():
                    cond_patterns = {
                        "coagulation_abnormal": "凝血|coagul",
                        "ulcer_history": "溃疡|ulcer",
                        "ckd": "肾|renal|ckd",
                        "liver_disease": "肝|liver|hepat",
                        "thrombocytopenia": "血小板|thrombocyto",
                    }
                    pattern = cond_patterns.get(cond_key, cond_key)
                    if re.search(pattern, disease, re.IGNORECASE):
                        scores.append(ConfoundingScore(
                            factor_name=cond_key,
                            dimension="baseline",
                            score=points,
                            evidence=f"MH2: {disease}",
                        ))
                        break

        return scores
