"""
AI-4: FieldAnnotator -- EDC column semantic role annotation.

Replaces the implicit assumption that field_0 = AE name, field_1 = start date,
etc. by using LLM to infer column meanings from data samples. Results are
cached so each form code only needs one annotation pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .engine import LLMEngine, LLMResponse

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "field_annotate.yaml"


@dataclass
class FieldAnnotation:
    """Semantic annotation for a set of EDC columns."""
    form_code: str
    annotations: dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    source: str = "ai"


# Well-known semantic roles that downstream modules rely on
KNOWN_ROLES = frozenset({
    "ae_name", "ae_start_date", "ae_end_date", "ae_severity", "ae_outcome",
    "ae_causality", "ae_action", "ae_sae_flag", "ae_description",
    "drug_name", "drug_indication", "cm_start_date", "cm_end_date",
    "cm_dose", "cm_unit", "cm_route", "cm_frequency",
    "test_code", "test_name", "result_value", "result_unit",
    "lower_limit", "upper_limit", "cs_flag", "collection_date",
    "vs_test_code", "vs_result", "vs_unit", "vs_date",
    "mh_term", "mh_start_date", "mh_end_date", "mh_ongoing", "mh_category",
    "birth_date", "sex", "race", "ethnicity", "enrollment_date",
    "disposition_status", "disposition_date", "disposition_reason",
    "subject_id", "visit_name", "form_status", "record_id",
    "free_text", "date_unknown", "numeric_unknown", "categorical_unknown",
})


class FieldAnnotator:
    """
    LLM-powered EDC column semantic annotator.

    Given a data sample from an EDC form, infers the semantic role of each
    field_n column. The annotation is cached per form_code for reuse.
    """

    SAMPLE_ROWS = 5

    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self._prompt = self._load_prompt()
        self._cache: dict[str, FieldAnnotation] = {}

    @staticmethod
    def _load_prompt() -> dict[str, Any]:
        if _PROMPT_PATH.exists():
            with open(_PROMPT_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def annotate(
        self,
        df: pd.DataFrame,
        form_code: str,
        form_name: str = "",
    ) -> FieldAnnotation:
        """
        Annotate field columns in an EDC table.

        Args:
            df: Standardized EDC table (columns: project_code, ..., field_0, field_1, ...).
            form_code: CRF form code (e.g., "AE", "CM", "LB1").
            form_name: Human-readable form name if available.

        Returns:
            FieldAnnotation with {column_name: semantic_role} mapping.
        """
        if form_code in self._cache:
            return self._cache[form_code]

        field_cols = [c for c in df.columns if c.startswith("field_")]
        if not field_cols:
            annotation = FieldAnnotation(form_code=form_code, confidence=0.0, source="no_fields")
            self._cache[form_code] = annotation
            return annotation

        annotation = self._apply_heuristics(df, field_cols, form_code)
        if annotation.confidence >= 0.8:
            self._cache[form_code] = annotation
            return annotation

        llm_annotation = self._llm_annotate(df, field_cols, form_code, form_name)
        merged = self._merge_annotations(annotation, llm_annotation)
        self._cache[form_code] = merged
        return merged

    def get_role(self, form_code: str, column: str) -> str | None:
        """Look up the semantic role of a column. Returns None if not annotated."""
        ann = self._cache.get(form_code)
        if ann is None:
            return None
        return ann.annotations.get(column)

    def find_column(self, form_code: str, role: str) -> str | None:
        """Find the column name for a given semantic role. Returns None if not found."""
        ann = self._cache.get(form_code)
        if ann is None:
            return None
        for col, r in ann.annotations.items():
            if r == role:
                return col
        return None

    def get_annotation_map(self, form_code: str) -> dict[str, str]:
        """Return the full column→role mapping for a form."""
        ann = self._cache.get(form_code)
        return ann.annotations.copy() if ann else {}

    # ---------- Heuristic annotation ----------

    def _apply_heuristics(
        self,
        df: pd.DataFrame,
        field_cols: list[str],
        form_code: str,
    ) -> FieldAnnotation:
        """Use form_code and simple data patterns for fast annotation."""
        annotations: dict[str, str] = {}
        fc = form_code.upper()

        if fc == "AE" and len(field_cols) >= 6:
            role_map = {
                "field_0": "ae_name",
                "field_1": "ae_start_date",
                "field_2": "ae_severity",
                "field_3": "ae_causality",
                "field_4": "ae_outcome",
                "field_5": "ae_sae_flag",
            }
            annotations.update({k: v for k, v in role_map.items() if k in field_cols})
            return FieldAnnotation(form_code=form_code, annotations=annotations, confidence=0.7, source="heuristic")

        if fc == "CM" and len(field_cols) >= 4:
            role_map = {
                "field_0": "drug_name",
                "field_1": "drug_indication",
                "field_2": "cm_start_date",
                "field_3": "cm_end_date",
            }
            annotations.update({k: v for k, v in role_map.items() if k in field_cols})
            return FieldAnnotation(form_code=form_code, annotations=annotations, confidence=0.7, source="heuristic")

        if fc.startswith("LB") and len(field_cols) >= 5:
            role_map = {
                "field_0": "test_code",
                "field_1": "test_name",
                "field_2": "result_value",
                "field_3": "lower_limit",
                "field_4": "upper_limit",
            }
            annotations.update({k: v for k, v in role_map.items() if k in field_cols})
            if len(field_cols) > 5:
                annotations["field_5"] = "cs_flag"
            return FieldAnnotation(form_code=form_code, annotations=annotations, confidence=0.65, source="heuristic")

        if fc in ("MH", "MH2") and len(field_cols) >= 3:
            role_map = {
                "field_0": "mh_term",
                "field_1": "mh_start_date",
                "field_2": "mh_end_date",
            }
            annotations.update({k: v for k, v in role_map.items() if k in field_cols})
            return FieldAnnotation(form_code=form_code, annotations=annotations, confidence=0.65, source="heuristic")

        return FieldAnnotation(form_code=form_code, annotations=annotations, confidence=0.0, source="none")

    # ---------- LLM annotation ----------

    def _llm_annotate(
        self,
        df: pd.DataFrame,
        field_cols: list[str],
        form_code: str,
        form_name: str,
    ) -> FieldAnnotation:
        """Use LLM to annotate columns based on data samples."""
        sample = df[field_cols].head(self.SAMPLE_ROWS).fillna("")
        sample_str = sample.to_string(index=False)

        system = self._prompt.get("system", "推断 EDC 列的语义角色。输出 JSON。")
        user_template = self._prompt.get("user_template", "")

        if user_template:
            user = user_template.format(
                form_code=form_code,
                form_name=form_name or "未知",
                column_names=", ".join(field_cols),
                sample_rows=sample_str,
            )
        else:
            user = (
                f"表单: {form_code} ({form_name or '未知'})\n"
                f"列名: {', '.join(field_cols)}\n"
                f"数据样本:\n{sample_str}"
            )

        try:
            resp = self.engine.complete(system=system, user=user, task_type="field_annotate")
            return self._parse_response(resp, form_code)
        except Exception as e:
            logger.warning("FieldAnnotator LLM call failed: %s", e)
            return FieldAnnotation(form_code=form_code, confidence=0.0, source="llm_error")

    @staticmethod
    def _parse_response(resp: LLMResponse, form_code: str) -> FieldAnnotation:
        data = resp.parse_json()
        if not isinstance(data, dict):
            return FieldAnnotation(form_code=form_code, confidence=0.0, source="parse_failed")

        annotations_raw = data.get("annotations", data)
        if "confidence" in annotations_raw:
            conf = float(annotations_raw.pop("confidence", 0.0))
        else:
            conf = float(data.get("confidence", 0.7))

        annotations = {}
        for col, role in annotations_raw.items():
            if isinstance(role, str) and (role in KNOWN_ROLES or role.startswith("field_")):
                annotations[col] = role
            elif isinstance(role, str):
                annotations[col] = role

        return FieldAnnotation(
            form_code=form_code,
            annotations=annotations,
            confidence=conf,
            source="ai",
        )

    @staticmethod
    def _merge_annotations(heuristic: FieldAnnotation, llm: FieldAnnotation) -> FieldAnnotation:
        """Merge heuristic and LLM annotations, preferring LLM when confident."""
        merged = heuristic.annotations.copy()

        for col, role in llm.annotations.items():
            if col not in merged or llm.confidence > heuristic.confidence:
                merged[col] = role

        return FieldAnnotation(
            form_code=heuristic.form_code,
            annotations=merged,
            confidence=max(heuristic.confidence, llm.confidence),
            source="merged",
        )
