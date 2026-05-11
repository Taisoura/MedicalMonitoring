"""
Module C: Subject-Level Integrated Database Builder

Builds a subject-level summary database by extracting and integrating
key variables from multiple EDC tables. Configurable dimensions allow
adaptation to different indications.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..parsers.edc_parser import EDCDataset
from ..utils.helpers import normalize_subject_id, safe_float


@dataclass
class SubjectDBConfig:
    """Configuration for subject-level database construction."""

    # Universal safety variables (always included)
    include_demographics: bool = True
    include_death: bool = True
    include_ae_summary: bool = True
    include_comorbidities: bool = True

    # Disease-specific scoring (configurable per indication)
    scoring_scales: list[dict[str, str]] = field(default_factory=list)

    # Comorbidity list to extract from MH2
    comorbidity_keywords: dict[str, str] = field(default_factory=lambda: {
        "hypertension": "高血压|Hypertension|HTN",
        "diabetes": "糖尿病|Diabetes|DM",
        "atrial_fibrillation": "房颤|Atrial Fibrillation|AF",
        "coronary_disease": "冠心病|Coronary|CAD|CHD",
        "hyperlipidemia": "高血脂|高脂血症|Hyperlipidemia",
    })

    # Treatment variables
    treatment_tables: list[str] = field(default_factory=list)

    # Endpoint events
    endpoint_tables: list[str] = field(default_factory=list)


# Pre-built configuration for stroke trials
STROKE_CONFIG = SubjectDBConfig(
    scoring_scales=[
        {"name": "nihss_v1", "table": "QS1", "visit_filter": "V1", "score_field": "field_19"},
        {"name": "nihss_v2", "table": "QS1", "visit_filter": "V2", "score_field": "field_19"},
        {"name": "nihss_v3", "table": "QS1", "visit_filter": "V3", "score_field": "field_19"},
        {"name": "nihss_v4", "table": "QS1", "visit_filter": "V4", "score_field": "field_19"},
        {"name": "mrs_v4", "table": "QS3", "visit_filter": "V4", "score_field": "field_3"},
        {"name": "mrs_v5", "table": "QS3", "visit_filter": "V5", "score_field": "field_3"},
        {"name": "aspects", "table": "QS4", "visit_filter": None, "score_field": "field_3"},
        {"name": "toast", "table": "QS5", "visit_filter": None, "score_field": "field_3"},
    ],
    treatment_tables=["PR1"],
    endpoint_tables=["CE1", "CE2", "CE3"],
    comorbidity_keywords={
        "hypertension": "高血压|Hypertension",
        "diabetes": "糖尿病|Diabetes",
        "atrial_fibrillation": "房颤|Atrial Fibrillation|AF",
        "coronary_disease": "冠心病|Coronary|CAD",
        "hyperlipidemia": "高血脂|高脂血症|Hyperlipidemia",
        "prior_stroke": "卒中|Stroke|脑梗",
    },
)


class SubjectDatabaseBuilder:
    """Builds a subject-level integrated database from EDC data."""

    def __init__(self, config: SubjectDBConfig | None = None):
        self.config = config or SubjectDBConfig()

    def build(self, dataset: EDCDataset) -> pd.DataFrame:
        """Build the subject-level database from all available tables."""
        subjects = dataset.subject_ids
        if not subjects:
            return pd.DataFrame()

        db = pd.DataFrame({"subject_id": subjects})

        if self.config.include_demographics:
            db = self._add_demographics(db, dataset)

        if self.config.include_death:
            db = self._add_death_info(db, dataset)

        if self.config.include_comorbidities:
            db = self._add_comorbidities(db, dataset)

        if self.config.include_ae_summary:
            db = self._add_ae_summary(db, dataset)

        for scale in self.config.scoring_scales:
            db = self._add_score(db, dataset, scale)

        for table_name in self.config.endpoint_tables:
            db = self._add_endpoint(db, dataset, table_name)

        db = self._compute_derived_variables(db)
        return db

    def _add_demographics(self, db: pd.DataFrame, dataset: EDCDataset) -> pd.DataFrame:
        dm_df = dataset.get_table("DM")
        if dm_df is None:
            return db

        demo_data = []
        for subj in db["subject_id"]:
            subj_rows = dm_df[dm_df["subject_id"].apply(normalize_subject_id) == subj]
            if subj_rows.empty:
                demo_data.append({"age": None, "sex": None})
                continue
            row = subj_rows.iloc[0]
            age = safe_float(row.get("field_2"))
            sex = str(row.get("field_3", "")).strip()
            demo_data.append({"age": age, "sex": sex})

        demo_df = pd.DataFrame(demo_data)
        return pd.concat([db.reset_index(drop=True), demo_df], axis=1)

    def _add_death_info(self, db: pd.DataFrame, dataset: EDCDataset) -> pd.DataFrame:
        dd_df = dataset.get_table("DD")
        if dd_df is None:
            db["death"] = 0
            db["cause_of_death"] = ""
            return db

        death_map = {}
        for _, row in dd_df.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            death_flag = str(row.get("field_0", "")).strip()
            if "是" in death_flag or "yes" in death_flag.lower():
                cause = str(row.get("field_3", "")).strip()
                death_map[subj] = cause

        db["death"] = db["subject_id"].apply(lambda s: 1 if s in death_map else 0)
        db["cause_of_death"] = db["subject_id"].apply(lambda s: death_map.get(s, ""))
        return db

    def _add_comorbidities(self, db: pd.DataFrame, dataset: EDCDataset) -> pd.DataFrame:
        mh_df = dataset.get_table("MH2")
        if mh_df is None:
            for name in self.config.comorbidity_keywords:
                db[name] = 0
            return db

        import re
        for comorb_name, pattern in self.config.comorbidity_keywords.items():
            comorb_subjects = set()
            for _, row in mh_df.iterrows():
                subj = normalize_subject_id(row.get("subject_id"))
                disease = str(row.get("field_1", "")).strip()
                if re.search(pattern, disease, re.IGNORECASE):
                    comorb_subjects.add(subj)
            db[comorb_name] = db["subject_id"].apply(
                lambda s: 1 if s in comorb_subjects else 0
            )
        return db

    def _add_ae_summary(self, db: pd.DataFrame, dataset: EDCDataset) -> pd.DataFrame:
        ae_df = dataset.get_table("AE")
        if ae_df is None:
            db["n_ae"] = 0
            db["n_sae"] = 0
            db["max_ae_grade"] = 0
            return db

        ae_counts = {}
        sae_counts = {}
        max_grades = {}

        for _, row in ae_df.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            if not subj:
                continue
            ae_counts[subj] = ae_counts.get(subj, 0) + 1

            sae_flag = str(row.get("field_5", "")).strip()
            if "是" in sae_flag or "yes" in sae_flag.lower():
                sae_counts[subj] = sae_counts.get(subj, 0) + 1

            grade = safe_float(row.get("field_2"))
            if grade is not None:
                max_grades[subj] = max(max_grades.get(subj, 0), grade)

        db["n_ae"] = db["subject_id"].apply(lambda s: ae_counts.get(s, 0))
        db["n_sae"] = db["subject_id"].apply(lambda s: sae_counts.get(s, 0))
        db["max_ae_grade"] = db["subject_id"].apply(lambda s: max_grades.get(s, 0))
        return db

    def _add_score(
        self, db: pd.DataFrame, dataset: EDCDataset, scale: dict[str, str]
    ) -> pd.DataFrame:
        table = dataset.get_table(scale["table"])
        if table is None:
            db[scale["name"]] = None
            return db

        score_map = {}
        for _, row in table.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            visit = str(row.get("visit", "")).strip()

            if scale.get("visit_filter"):
                if scale["visit_filter"].upper() not in visit.upper():
                    continue

            score_val = safe_float(row.get(scale["score_field"]))
            if score_val is not None:
                score_map[subj] = score_val

        db[scale["name"]] = db["subject_id"].apply(lambda s: score_map.get(s))
        return db

    def _add_endpoint(
        self, db: pd.DataFrame, dataset: EDCDataset, table_name: str
    ) -> pd.DataFrame:
        table = dataset.get_table(table_name)
        col_name = table_name.lower()

        if table is None:
            db[col_name] = 0
            return db

        event_subjects = set()
        for _, row in table.iterrows():
            subj = normalize_subject_id(row.get("subject_id"))
            event_flag = str(row.get("field_0", "")).strip()
            if "是" in event_flag or "yes" in event_flag.lower():
                event_subjects.add(subj)

        db[col_name] = db["subject_id"].apply(lambda s: 1 if s in event_subjects else 0)
        return db

    def _compute_derived_variables(self, db: pd.DataFrame) -> pd.DataFrame:
        """Compute derived variables like score changes."""
        if "nihss_v1" in db.columns and "nihss_v2" in db.columns:
            db["delta_nihss_v1v2"] = db.apply(
                lambda r: (r["nihss_v2"] - r["nihss_v1"])
                if pd.notna(r.get("nihss_v1")) and pd.notna(r.get("nihss_v2"))
                else None,
                axis=1,
            )

        if "nihss_v1" in db.columns:
            db["nihss_worsening"] = db["delta_nihss_v1v2"].apply(
                lambda x: 1 if x is not None and x >= 10 else 0
            ) if "delta_nihss_v1v2" in db.columns else 0

        return db
