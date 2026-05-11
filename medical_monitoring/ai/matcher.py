"""
AI-2: SemanticMatcher -- cross-domain concept alignment.

Replaces brittle regex patterns in cross_domain.py and ctcae_grading.py
with LLM-powered semantic matching for:
  - CM → AE treatment relationship
  - LB abnormality → AE correspondence
  - CTCAE term → AE record alignment
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .engine import LLMEngine, LLMResponse

logger = logging.getLogger(__name__)

_CM_AE_PROMPT = Path(__file__).parent / "prompts" / "match_cm_ae.yaml"
_LB_AE_PROMPT = Path(__file__).parent / "prompts" / "match_lb_ae.yaml"


@dataclass
class MatchResult:
    """Result of a semantic matching operation."""
    matches: bool
    confidence: float = 0.0
    reasoning: str = ""
    mechanism: str = ""
    source: str = "ai"


class SemanticMatcher:
    """
    LLM-powered cross-domain semantic matcher.

    Designed as a fallback/replacement for rule-based matching in:
      - CrossDomainAuditor._cm_treats_ae()
      - CrossDomainAuditor._find_matching_ae_for_lab()
      - CTCAEGradingEngine._find_matching_ae()
    """

    BATCH_SIZE = 15

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._cm_ae_prompt = self._load_prompt(_CM_AE_PROMPT)
        self._lb_ae_prompt = self._load_prompt(_LB_AE_PROMPT)

    @staticmethod
    def _load_prompt(path: Path) -> dict[str, Any]:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    # ---------- CM-AE matching ----------

    def cm_treats_ae(
        self,
        drug_name: str,
        indication: str,
        ae_name: str,
        cm_start: str = "",
        ae_start: str = "",
    ) -> MatchResult:
        """Determine if a concomitant medication treats the given AE."""
        system = self._cm_ae_prompt.get("system", "判断 CM 是否治疗 AE。输出 JSON。")
        user = (
            f"CM药物: {drug_name}\n"
            f"CM适应症: {indication}\n"
            f"AE名称: {ae_name}\n"
            f"CM开始日期: {cm_start}\n"
            f"AE开始日期: {ae_start}"
        )

        try:
            resp = self.engine.complete(system=system, user=user, task_type="match_cm_ae")
            return self._parse_match_response(resp)
        except Exception as e:
            logger.warning("CM-AE match failed: %s", e)
            return MatchResult(matches=False, confidence=0.0, reasoning=str(e))

    def batch_cm_treats_ae(
        self,
        pairs: list[dict[str, str]],
    ) -> list[MatchResult]:
        """
        Batch evaluate CM-AE treatment relationships.

        Each pair: {drug_name, indication, ae_name, cm_start?, ae_start?}
        """
        if not pairs:
            return []

        all_results: list[MatchResult] = []
        for i in range(0, len(pairs), self.BATCH_SIZE):
            batch = pairs[i : i + self.BATCH_SIZE]
            batch_results = self._batch_cm_ae(batch)
            all_results.extend(batch_results)
        return all_results

    def _batch_cm_ae(self, pairs: list[dict[str, str]]) -> list[MatchResult]:
        system = self._cm_ae_prompt.get("system", "判断 CM 是否治疗 AE。输出 JSON 数组。")
        batch_template = self._cm_ae_prompt.get("batch_user_template", "")

        if batch_template:
            rows = []
            for idx, p in enumerate(pairs):
                rows.append(
                    f"| {idx} | {p.get('drug_name','')} | {p.get('indication','')} "
                    f"| {p.get('ae_name','')} | {p.get('cm_start','')} | {p.get('ae_start','')} |"
                )
            user_msg = batch_template.format(table_rows="\n".join(rows))
        else:
            lines = []
            for idx, p in enumerate(pairs):
                lines.append(
                    f"{idx}. CM: {p.get('drug_name','')} ({p.get('indication','')}) → AE: {p.get('ae_name','')}"
                )
            user_msg = "请判断以下 CM-AE 对的治疗关系：\n" + "\n".join(lines)

        try:
            resp = self.engine.complete(system=system, user=user_msg, task_type="match_cm_ae_batch")
            return self._parse_batch_match(resp, len(pairs))
        except Exception as e:
            logger.warning("Batch CM-AE match failed: %s", e)
            return [MatchResult(matches=False, confidence=0.0, reasoning=str(e)) for _ in pairs]

    # ---------- LB-AE matching ----------

    def lab_matches_ae(
        self,
        test_name: str,
        test_code: str,
        result_value: str,
        ref_range: str,
        ae_name: str,
        ae_start: str = "",
        lab_date: str = "",
    ) -> MatchResult:
        """Determine if a lab abnormality corresponds to a specific AE."""
        system = self._lb_ae_prompt.get("system", "判断 LB 异常是否对应 AE。输出 JSON。")
        user = (
            f"实验室检查项: {test_name} ({test_code})\n"
            f"检测值: {result_value} | 参考范围: {ref_range}\n"
            f"AE名称: {ae_name}\n"
            f"LB检测日期: {lab_date} | AE开始日期: {ae_start}"
        )

        try:
            resp = self.engine.complete(system=system, user=user, task_type="match_lb_ae")
            return self._parse_match_response(resp)
        except Exception as e:
            logger.warning("LB-AE match failed: %s", e)
            return MatchResult(matches=False, confidence=0.0, reasoning=str(e))

    def batch_lab_matches_ae(
        self,
        pairs: list[dict[str, str]],
    ) -> list[MatchResult]:
        """Batch evaluate LB-AE correspondences."""
        if not pairs:
            return []

        all_results: list[MatchResult] = []
        for i in range(0, len(pairs), self.BATCH_SIZE):
            batch = pairs[i : i + self.BATCH_SIZE]

            system = self._lb_ae_prompt.get("system", "")
            user_template = self._lb_ae_prompt.get("user_template", "")
            rows = []
            for idx, p in enumerate(batch):
                rows.append(
                    f"| {idx} | {p.get('test_name','')} | {p.get('result_value','')} "
                    f"| {p.get('ref_range','')} | {p.get('direction','')} "
                    f"| {p.get('ae_name','')} | {p.get('ae_start','')} | {p.get('lab_date','')} |"
                )
            user_msg = user_template.format(table_rows="\n".join(rows)) if user_template else "\n".join(rows)

            try:
                resp = self.engine.complete(system=system, user=user_msg, task_type="match_lb_ae_batch")
                all_results.extend(self._parse_batch_match(resp, len(batch)))
            except Exception as e:
                logger.warning("Batch LB-AE match failed: %s", e)
                all_results.extend(MatchResult(matches=False, confidence=0.0, reasoning=str(e)) for _ in batch)

        return all_results

    # ---------- CTCAE-AE alignment ----------

    def ctcae_matches_ae(
        self,
        ctcae_term: str,
        lab_abnormality: str,
        ae_name: str,
    ) -> MatchResult:
        """Determine if a CTCAE term / lab abnormality matches an AE record."""
        system = (
            "你是 CTCAE v5.0 和 MedDRA 编码专家。判断以下实验室异常/CTCAE术语是否与AE记录对应。\n"
            "输出 JSON: {\"matches\": bool, \"confidence\": float, \"reasoning\": str}"
        )
        user = (
            f"CTCAE 术语: {ctcae_term}\n"
            f"实验室异常: {lab_abnormality}\n"
            f"AE 记录名称: {ae_name}"
        )

        try:
            resp = self.engine.complete(system=system, user=user, task_type="match_ctcae_ae")
            return self._parse_match_response(resp)
        except Exception as e:
            logger.warning("CTCAE-AE match failed: %s", e)
            return MatchResult(matches=False, confidence=0.0, reasoning=str(e))

    # ---------- Generic matching ----------

    def semantic_match(
        self,
        concept_a: str,
        concept_b: str,
        context: str = "clinical trial",
    ) -> MatchResult:
        """Generic semantic similarity/relationship check between two medical concepts."""
        system = (
            "你是医学术语专家。判断以下两个医学概念是否指代相同或密切相关的临床状况。\n"
            "输出 JSON: {\"matches\": bool, \"confidence\": float, \"reasoning\": str}"
        )
        user = f"概念 A: {concept_a}\n概念 B: {concept_b}\n上下文: {context}"

        try:
            resp = self.engine.complete(system=system, user=user, task_type="semantic_match")
            return self._parse_match_response(resp)
        except Exception as e:
            logger.warning("Semantic match failed: %s", e)
            return MatchResult(matches=False, confidence=0.0, reasoning=str(e))

    # ---------- Response parsing ----------

    @staticmethod
    def _parse_match_response(resp: LLMResponse) -> MatchResult:
        data = resp.parse_json()
        if data is None:
            return MatchResult(matches=False, confidence=0.0, reasoning="JSON parse failed")
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            return MatchResult(matches=False, confidence=0.0, reasoning="unexpected format")

        return MatchResult(
            matches=bool(data.get("matches", data.get("treats_ae", False))),
            confidence=float(data.get("confidence", 0.0)),
            reasoning=data.get("reasoning", ""),
            mechanism=data.get("mechanism", data.get("relationship", "")),
            source="ai",
        )

    @staticmethod
    def _parse_batch_match(resp: LLMResponse, expected: int) -> list[MatchResult]:
        data = resp.parse_json()
        if data is None or not isinstance(data, list):
            return [MatchResult(matches=False, confidence=0.0, reasoning="parse failed")] * expected

        results = []
        for item in data:
            if not isinstance(item, dict):
                results.append(MatchResult(matches=False, confidence=0.0))
                continue
            results.append(MatchResult(
                matches=bool(item.get("matches", item.get("treats_ae", False))),
                confidence=float(item.get("confidence", 0.0)),
                reasoning=item.get("reasoning", ""),
                mechanism=item.get("mechanism", item.get("relationship", "")),
                source="ai",
            ))

        while len(results) < expected:
            results.append(MatchResult(matches=False, confidence=0.0, reasoning="no response"))
        return results[:expected]
