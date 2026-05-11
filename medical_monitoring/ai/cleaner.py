"""
AI-5: DataCleaner -- typo correction, format unification, negation detection,
and missing-value inference for clinical trial data.

This is the first step in the AI pipeline: cleaning raw data before
normalization and semantic matching improves all downstream accuracy.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .engine import LLMEngine, LLMResponse

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "clean_data.yaml"


@dataclass
class CleanResult:
    """Result of cleaning a single data field."""
    original: str
    corrected: str
    correction_type: str  # typo | format | negation | inference | none
    confidence: float = 0.0
    explanation: str = ""
    not_present: bool = False


# Severity format synonyms for rule-based fast path
_SEVERITY_MAP: dict[str, str] = {
    "i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
    "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
    "一级": "1", "二级": "2", "三级": "3", "四级": "4", "五级": "5",
    "1级": "1", "2级": "2", "3级": "3", "4级": "4", "5级": "5",
    "轻度": "1", "中度": "2", "重度": "3", "严重": "3",
    "危及生命": "4", "死亡": "5",
    "grade 1": "1", "grade 2": "2", "grade 3": "3", "grade 4": "4", "grade 5": "5",
    "Grade 1": "1", "Grade 2": "2", "Grade 3": "3", "Grade 4": "4", "Grade 5": "5",
    "mild": "1", "moderate": "2", "severe": "3", "life-threatening": "4",
}

_NEGATION_PATTERNS = re.compile(
    r"(?:^|[，,；;。\s])"
    r"(?:排除|无|否认|没有|不伴|未见|未发现|未诉|非|不考虑)"
    r"(.+)",
    re.IGNORECASE,
)


class DataCleaner:
    """
    LLM-augmented clinical data cleaner.

    Applies fast rule-based cleaning first, then escalates ambiguous cases
    to the LLM for typo correction and context-dependent inference.
    """

    BATCH_SIZE = 40

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> dict[str, Any]:
        if _PROMPT_PATH.exists():
            with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def clean_severity(self, values: list[str]) -> list[CleanResult]:
        """Fast-path severity format unification (no LLM needed)."""
        results = []
        for v in values:
            stripped = v.strip()
            normalized = _SEVERITY_MAP.get(stripped)
            if normalized is None:
                m = re.search(r"(\d)", stripped)
                if m:
                    normalized = m.group(1)
            if normalized and normalized != stripped:
                results.append(CleanResult(
                    original=stripped, corrected=normalized,
                    correction_type="format", confidence=1.0,
                    explanation=f"标准化严重度格式: '{stripped}' → '{normalized}'",
                ))
            else:
                results.append(CleanResult(
                    original=stripped, corrected=stripped,
                    correction_type="none", confidence=1.0,
                ))
        return results

    def detect_negations(self, texts: list[str]) -> list[CleanResult]:
        """Detect negation expressions in medical text (rule-based)."""
        results = []
        for t in texts:
            m = _NEGATION_PATTERNS.search(t)
            if m:
                results.append(CleanResult(
                    original=t, corrected=m.group(1).strip(),
                    correction_type="negation", confidence=0.9,
                    explanation=f"否定表达检测: '{t}'",
                    not_present=True,
                ))
            else:
                results.append(CleanResult(
                    original=t, corrected=t,
                    correction_type="none", confidence=1.0,
                ))
        return results

    def clean_batch(
        self,
        values: list[str],
        field_type: str = "general",
        context: str = "",
    ) -> list[CleanResult]:
        """
        Clean a batch of values using combined rule + LLM approach.

        Args:
            values: Raw data values to clean.
            field_type: Type hint ('ae_name', 'date', 'severity', 'drug_name', 'general').
            context: Additional context for the LLM (e.g., form name).
        """
        if field_type == "severity":
            return self.clean_severity(values)

        rule_results = self._rule_based_clean(values, field_type)

        ambiguous_indices = [
            i for i, r in enumerate(rule_results)
            if r.correction_type == "none" and self._needs_llm(r.original, field_type)
        ]

        if not ambiguous_indices:
            return rule_results

        ambiguous_values = [values[i] for i in ambiguous_indices]
        llm_results = self._llm_clean_batch(ambiguous_values, field_type, context)

        for idx, llm_result in zip(ambiguous_indices, llm_results):
            if llm_result.correction_type != "none" and llm_result.confidence > 0.6:
                rule_results[idx] = llm_result

        return rule_results

    def _rule_based_clean(self, values: list[str], field_type: str) -> list[CleanResult]:
        """Apply fast rule-based cleaning."""
        results = []
        for v in values:
            stripped = v.strip()
            result = CleanResult(original=stripped, corrected=stripped, correction_type="none", confidence=1.0)

            if not stripped or stripped.lower() in ("nan", "none", "null", "na", "n/a"):
                result.corrected = ""
                result.correction_type = "format"
                result.explanation = "空值标准化"
                results.append(result)
                continue

            neg = _NEGATION_PATTERNS.search(stripped)
            if neg:
                result.corrected = neg.group(1).strip()
                result.correction_type = "negation"
                result.confidence = 0.9
                result.not_present = True
                result.explanation = f"否定表达: '{stripped}'"
                results.append(result)
                continue

            if field_type == "date":
                cleaned = re.sub(r"[年月]", "-", stripped).rstrip("日号")
                if cleaned != stripped:
                    result.corrected = cleaned
                    result.correction_type = "format"
                    result.explanation = "日期格式标准化"
                results.append(result)
                continue

            results.append(result)
        return results

    @staticmethod
    def _needs_llm(value: str, field_type: str) -> bool:
        """Heuristic: does this value likely contain a typo or need inference?"""
        if not value or len(value) < 2:
            return False
        if field_type in ("ae_name", "drug_name", "mh_term"):
            return True
        if "UK" in value or "UNK" in value.upper():
            return True
        if re.search(r"[a-zA-Z]", value) and re.search(r"[\u4e00-\u9fff]", value):
            return True
        return False

    def _llm_clean_batch(
        self,
        values: list[str],
        field_type: str,
        context: str,
    ) -> list[CleanResult]:
        """Escalate ambiguous values to LLM for cleaning."""
        system = self._prompt.get("system", "Clean clinical data fields. Output JSON array.")
        user_template = self._prompt.get("user_template", "请清洗：\n{data_rows}")

        all_results: list[CleanResult] = []
        for i in range(0, len(values), self.BATCH_SIZE):
            batch = values[i : i + self.BATCH_SIZE]
            data_rows = "\n".join(f"{j}. {v}" for j, v in enumerate(batch))
            user_msg = user_template.format(
                field_type=field_type,
                context=context or "临床试验 EDC 数据",
                data_rows=data_rows,
            )
            try:
                resp = self.engine.complete(
                    system=system, user=user_msg, task_type=f"clean_{field_type}",
                )
                all_results.extend(self._parse_response(resp, batch))
            except Exception as e:
                logger.warning("DataCleaner LLM batch failed: %s", e)
                all_results.extend(
                    CleanResult(original=v, corrected=v, correction_type="none",
                                confidence=0.0, explanation=f"LLM error: {e}")
                    for v in batch
                )
        return all_results

    @staticmethod
    def _parse_response(resp: LLMResponse, originals: list[str]) -> list[CleanResult]:
        data = resp.parse_json()
        if data is None:
            return [CleanResult(original=v, corrected=v, correction_type="none",
                                confidence=0.0, explanation="parse failed") for v in originals]

        if isinstance(data, dict):
            data = [data]

        results = []
        for item in data:
            if not isinstance(item, dict):
                continue
            ctype = item.get("correction_type", "none")
            results.append(CleanResult(
                original=item.get("original", ""),
                corrected=item.get("corrected_value", item.get("corrected", item.get("original", ""))),
                correction_type=ctype,
                confidence=float(item.get("confidence", 0.0)),
                explanation=item.get("explanation", ""),
                not_present=ctype == "negation",
            ))

        while len(results) < len(originals):
            idx = len(results)
            results.append(CleanResult(
                original=originals[idx], corrected=originals[idx],
                correction_type="none", confidence=0.0,
            ))
        return results[:len(originals)]
