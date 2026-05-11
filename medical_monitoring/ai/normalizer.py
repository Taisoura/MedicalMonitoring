"""
AI-1: MedTermNormalizer -- medical terminology normalization.

Maps free-text AE/CM/MH terms to MedDRA Preferred Terms (PT) via LLM,
replacing the limited regex-based NAMING_RULES in ae_quality.py.
Results are cached in SQLite for high hit rates within a trial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .engine import LLMEngine, LLMResponse

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "normalize_ae.yaml"


@dataclass
class NormalizedTerm:
    """Result of normalizing a single medical term."""
    original: str
    pt_en: str = ""
    pt_cn: str = ""
    llt: str = ""
    soc: str = ""
    confidence: float = 0.0
    note: str = ""
    source: str = "ai"


class MedTermNormalizer:
    """
    LLM-powered medical terminology normalizer.

    Batch-processes AE/CM/MH free-text terms and maps them to MedDRA PTs.
    Uses prompt templates from prompts/normalize_ae.yaml.
    """

    BATCH_SIZE = 30

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> dict[str, Any]:
        if _PROMPT_PATH.exists():
            with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def normalize_terms(
        self,
        terms: list[str],
        term_type: str = "ae",
    ) -> list[NormalizedTerm]:
        """
        Normalize a list of medical terms to MedDRA PTs.

        Args:
            terms: Raw free-text terms (e.g., AE names from EDC).
            term_type: One of 'ae', 'cm', 'mh' for prompt context.

        Returns:
            List of NormalizedTerm objects (may be longer than input if
            composite terms are split).
        """
        unique_terms = list(dict.fromkeys(t.strip() for t in terms if t.strip()))
        if not unique_terms:
            return []

        all_results: list[NormalizedTerm] = []
        for i in range(0, len(unique_terms), self.BATCH_SIZE):
            batch = unique_terms[i : i + self.BATCH_SIZE]
            batch_results = self._normalize_batch(batch, term_type)
            all_results.extend(batch_results)

        term_map: dict[str, list[NormalizedTerm]] = {}
        for r in all_results:
            term_map.setdefault(r.original, []).append(r)

        output: list[NormalizedTerm] = []
        for t in terms:
            stripped = t.strip()
            if stripped in term_map:
                output.extend(term_map[stripped])
            else:
                output.append(NormalizedTerm(original=stripped, confidence=0.0, note="no match"))

        return output

    def normalize_single(self, term: str, term_type: str = "ae") -> list[NormalizedTerm]:
        """Normalize a single term (may return multiple if composite)."""
        return self.normalize_terms([term], term_type)

    def build_variant_groups(
        self,
        normalized: list[NormalizedTerm],
    ) -> dict[str, list[NormalizedTerm]]:
        """
        Group normalized terms by PT to detect naming variants.

        Returns:
            {pt_en: [NormalizedTerm, ...]} where len > 1 indicates variants.
        """
        groups: dict[str, list[NormalizedTerm]] = {}
        for n in normalized:
            if n.pt_en:
                groups.setdefault(n.pt_en, []).append(n)
        return {k: v for k, v in groups.items() if len(v) > 1}

    def _normalize_batch(
        self,
        terms: list[str],
        term_type: str,
    ) -> list[NormalizedTerm]:
        """Send a batch of terms to the LLM for normalization."""
        system_prompt = self._prompt.get("system", "Normalize medical terms to MedDRA PT. Output JSON array.")
        user_template = self._prompt.get("user_template", "请归一化以下术语：\n{terms}")
        user_msg = user_template.format(terms="\n".join(f"- {t}" for t in terms))

        try:
            resp = self.engine.complete(
                system=system_prompt,
                user=user_msg,
                task_type=f"normalize_{term_type}",
            )
            return self._parse_response(resp, terms)
        except Exception as e:
            logger.warning("MedTermNormalizer batch failed: %s", e)
            return [NormalizedTerm(original=t, confidence=0.0, note=f"LLM error: {e}") for t in terms]

    @staticmethod
    def _parse_response(resp: LLMResponse, original_terms: list[str]) -> list[NormalizedTerm]:
        """Parse LLM JSON response into NormalizedTerm objects."""
        data = resp.parse_json()
        if data is None:
            return [NormalizedTerm(original=t, confidence=0.0, note="JSON parse failed") for t in original_terms]

        if isinstance(data, dict):
            data = [data]

        results = []
        for item in data:
            if not isinstance(item, dict):
                continue
            results.append(NormalizedTerm(
                original=item.get("original", ""),
                pt_en=item.get("pt_en", item.get("pt", "")),
                pt_cn=item.get("pt_cn", ""),
                llt=item.get("llt", ""),
                soc=item.get("soc", ""),
                confidence=float(item.get("confidence", 0.0)),
                note=item.get("note", ""),
                source="ai",
            ))

        if not results:
            return [NormalizedTerm(original=t, confidence=0.0, note="empty response") for t in original_terms]

        return results
