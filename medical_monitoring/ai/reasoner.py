"""
AI-3: ClinicalReasoner -- LLM-powered clinical reasoning support.

Enhances causality assessment, SAE screening, and severity reasonability
checks with structured clinical reasoning instead of simple threshold logic.

IMPORTANT: AI outputs are marked as "AI-suggested" and serve as pre-screening
recommendations only -- final judgment authority rests with the medical monitor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .engine import LLMEngine, LLMResponse

logger = logging.getLogger(__name__)

_CAUSALITY_PROMPT = Path(__file__).parent / "prompts" / "causality_assess.yaml"


@dataclass
class CausalityAssessment:
    """Structured causality assessment from AI reasoning."""
    ae_name: str
    subject_id: str = ""
    temporal_relationship: str = ""
    biological_plausibility: str = ""
    dechallenge: str = ""
    rechallenge: str = ""
    confounding_factors: list[str] | None = None
    alternative_explanations: list[str] | None = None
    suggested_category: str = ""  # certain|probable|possible|unlikely|conditional|unassessable
    reasoning: str = ""
    confidence: float = 0.0
    key_evidence: str = ""
    source: str = "AI-suggested"

    def __post_init__(self):
        if self.confounding_factors is None:
            self.confounding_factors = []
        if self.alternative_explanations is None:
            self.alternative_explanations = []


@dataclass
class SAEScreening:
    """AI assessment of whether an AE should be flagged as SAE."""
    ae_name: str
    subject_id: str = ""
    should_be_sae: bool = False
    sae_criteria_met: list[str] | None = None
    reasoning: str = ""
    confidence: float = 0.0
    source: str = "AI-suggested"

    def __post_init__(self):
        if self.sae_criteria_met is None:
            self.sae_criteria_met = []


@dataclass
class SeverityAssessment:
    """AI assessment of AE severity/grade reasonability."""
    ae_name: str
    reported_grade: int | None = None
    suggested_grade: int | None = None
    discrepancy: bool = False
    reasoning: str = ""
    confidence: float = 0.0
    source: str = "AI-suggested"


class ClinicalReasoner:
    """
    LLM-powered clinical reasoning engine.

    Provides structured assessments for:
      - Drug-AE causality (WHO-UMC criteria)
      - SAE screening (ICH-E2A criteria)
      - Severity/grade reasonability (CTCAE v5.0)
    """

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._causality_prompt = self._load_prompt(_CAUSALITY_PROMPT)

    @staticmethod
    def _load_prompt(path: Path) -> dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ---------- Causality Assessment ----------

    def assess_causality(
        self,
        ae_name: str,
        drug_name: str = "",
        indication: str = "",
        known_risks: str = "",
        severity: str = "",
        ae_start: str = "",
        ae_end: str = "",
        drug_start: str = "",
        drug_end: str = "",
        investigator_causality: str = "",
        medical_history: str = "",
        concomitant_meds: str = "",
        subject_id: str = "",
    ) -> CausalityAssessment:
        """
        Generate a structured causality assessment for a single AE.

        Automatically adapts to information availability:
        - If drug_name + known_risks provided: full assessment
        - If drug_name only: partial assessment (drug class inference)
        - If no drug info: blind mode (data-driven analysis only)
        """
        system = self._causality_prompt.get("system", "评估 AE 与药物的因果关系。输出 JSON。")

        is_blind = not drug_name or drug_name in ("", "（未提供）", "(not provided)")
        has_risks = bool(known_risks and known_risks.strip())

        if is_blind:
            template = self._causality_prompt.get("user_template_blind", "")
            if template:
                user = template.format(
                    ae_name=ae_name,
                    severity=severity,
                    ae_start=ae_start,
                    ae_end=ae_end,
                    investigator_causality=investigator_causality,
                    medical_history=medical_history,
                    concomitant_meds=concomitant_meds,
                )
            else:
                user = (
                    f"AE: {ae_name} (严重度: {severity})\n"
                    f"AE时间: {ae_start} ~ {ae_end}\n"
                    f"研究者判断: {investigator_causality}\n"
                    f"病史: {medical_history}\n合并用药: {concomitant_meds}\n"
                    f"注意: 研究药物信息因涉密未提供，请基于数据驱动分析。"
                )
        else:
            user_template = self._causality_prompt.get("user_template", "")
            if user_template:
                user = user_template.format(
                    drug_name=drug_name,
                    indication=indication or "（未提供）",
                    known_risks=known_risks or "（未提供 -- IB 涉密或未获取）",
                    ae_name=ae_name,
                    severity=severity,
                    ae_start=ae_start,
                    ae_end=ae_end,
                    drug_start=drug_start,
                    drug_end=drug_end,
                    investigator_causality=investigator_causality,
                    medical_history=medical_history,
                    concomitant_meds=concomitant_meds,
                )
            else:
                user = (
                    f"药物: {drug_name} ({indication})\n"
                    f"IB已知风险: {known_risks or '（未提供）'}\n"
                    f"AE: {ae_name} (严重度: {severity})\n"
                    f"AE时间: {ae_start} ~ {ae_end}\n药物时间: {drug_start} ~ {drug_end}\n"
                    f"研究者判断: {investigator_causality}\n"
                    f"病史: {medical_history}\n合并用药: {concomitant_meds}"
                )

        try:
            resp = self.engine.complete(system=system, user=user, task_type="causality_assess")
            return self._parse_causality(resp, ae_name, subject_id)
        except Exception as e:
            logger.warning("Causality assessment failed: %s", e)
            return CausalityAssessment(
                ae_name=ae_name, subject_id=subject_id,
                reasoning=f"LLM error: {e}", confidence=0.0,
            )

    def batch_assess_causality(
        self,
        cases: list[dict[str, str]],
    ) -> list[CausalityAssessment]:
        """Assess causality for multiple AE cases."""
        return [self.assess_causality(**case) for case in cases]

    # ---------- SAE Screening ----------

    def screen_sae(
        self,
        ae_name: str,
        grade: int | None = None,
        outcome: str = "",
        hospitalization: str = "",
        medical_history: str = "",
        subject_id: str = "",
    ) -> SAEScreening:
        """
        Evaluate whether an AE should be reported as SAE per ICH-E2A criteria.

        ICH-E2A SAE criteria:
        - Death
        - Life-threatening
        - Hospitalization (initial or prolonged)
        - Persistent/significant disability
        - Congenital anomaly
        - Important medical event
        """
        system = (
            "你是药物安全专家。基于 ICH-E2A 标准评估以下 AE 是否符合 SAE 标准。\n"
            "SAE 标准: 死亡、危及生命、住院（含延长住院）、持续或显著残疾/功能障碍、"
            "先天性异常/出生缺陷、重要医学事件。\n"
            "输出 JSON: {\"should_be_sae\": bool, \"sae_criteria_met\": [str], "
            "\"reasoning\": str, \"confidence\": float}\n"
            "仅输出 JSON。"
        )
        user = (
            f"AE名称: {ae_name}\n"
            f"严重度等级: Grade {grade}\n"
            f"转归: {outcome}\n"
            f"住院情况: {hospitalization}\n"
            f"既往病史: {medical_history}"
        )

        try:
            resp = self.engine.complete(system=system, user=user, task_type="sae_screen")
            return self._parse_sae(resp, ae_name, subject_id)
        except Exception as e:
            logger.warning("SAE screening failed: %s", e)
            return SAEScreening(
                ae_name=ae_name, subject_id=subject_id,
                reasoning=f"LLM error: {e}", confidence=0.0,
            )

    # ---------- Severity Reasonability ----------

    def assess_severity(
        self,
        ae_name: str,
        reported_grade: int | None = None,
        ae_description: str = "",
        functional_impact: str = "",
    ) -> SeverityAssessment:
        """
        Evaluate whether the reported CTCAE grade is reasonable for the given AE.
        """
        system = (
            "你是 CTCAE v5.0 分级专家。评估以下 AE 的报告等级是否合理。\n"
            "考虑: AE 名称在 CTCAE 中的分级定义、功能影响描述、是否存在 CTCAE 中"
            "该 AE 不允许的低等级。\n"
            "输出 JSON: {\"suggested_grade\": int, \"discrepancy\": bool, "
            "\"reasoning\": str, \"confidence\": float}\n"
            "仅输出 JSON。"
        )
        user = (
            f"AE名称: {ae_name}\n"
            f"报告等级: Grade {reported_grade}\n"
            f"AE描述: {ae_description}\n"
            f"功能影响: {functional_impact}"
        )

        try:
            resp = self.engine.complete(system=system, user=user, task_type="severity_assess")
            return self._parse_severity(resp, ae_name, reported_grade)
        except Exception as e:
            logger.warning("Severity assessment failed: %s", e)
            return SeverityAssessment(
                ae_name=ae_name, reported_grade=reported_grade,
                reasoning=f"LLM error: {e}", confidence=0.0,
            )

    # ---------- Response parsers ----------

    @staticmethod
    def _parse_causality(resp: LLMResponse, ae_name: str, subject_id: str) -> CausalityAssessment:
        data = resp.parse_json()
        if not isinstance(data, dict):
            return CausalityAssessment(
                ae_name=ae_name, subject_id=subject_id,
                reasoning="parse failed", confidence=0.0,
            )
        return CausalityAssessment(
            ae_name=ae_name,
            subject_id=subject_id,
            temporal_relationship=data.get("temporal_relationship", ""),
            biological_plausibility=data.get("biological_plausibility", ""),
            dechallenge=data.get("dechallenge", ""),
            rechallenge=data.get("rechallenge", ""),
            confounding_factors=data.get("confounding_factors", []),
            alternative_explanations=data.get("alternative_explanations", []),
            suggested_category=data.get("suggested_category", ""),
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.0)),
            key_evidence=data.get("key_evidence", ""),
        )

    @staticmethod
    def _parse_sae(resp: LLMResponse, ae_name: str, subject_id: str) -> SAEScreening:
        data = resp.parse_json()
        if not isinstance(data, dict):
            return SAEScreening(
                ae_name=ae_name, subject_id=subject_id,
                reasoning="parse failed", confidence=0.0,
            )
        return SAEScreening(
            ae_name=ae_name,
            subject_id=subject_id,
            should_be_sae=bool(data.get("should_be_sae", False)),
            sae_criteria_met=data.get("sae_criteria_met", []),
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.0)),
        )

    @staticmethod
    def _parse_severity(resp: LLMResponse, ae_name: str, reported: int | None) -> SeverityAssessment:
        data = resp.parse_json()
        if not isinstance(data, dict):
            return SeverityAssessment(
                ae_name=ae_name, reported_grade=reported,
                reasoning="parse failed", confidence=0.0,
            )
        suggested = data.get("suggested_grade")
        try:
            suggested = int(suggested) if suggested is not None else None
        except (ValueError, TypeError):
            suggested = None

        return SeverityAssessment(
            ae_name=ae_name,
            reported_grade=reported,
            suggested_grade=suggested,
            discrepancy=bool(data.get("discrepancy", False)),
            reasoning=data.get("reasoning", ""),
            confidence=float(data.get("confidence", 0.0)),
        )
