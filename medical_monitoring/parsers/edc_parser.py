"""
Module A: EDC Data Parser

Parses multi-sheet EDC export Excel files into a standardized data structure.
Supports two formats:
  1. Generic EDC: 14-column header (项目编号/表单编号/受试者编号...)
  2. CDISC EDC: Two-row header (Row 0 = Chinese labels, Row 1 = CDISC variable codes)
Auto-detects format and normalises both into a unified EDCDataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# ── Generic EDC (non-CDISC) 14-column structure ─────────────────────────────

EDC_STANDARD_COLUMNS = [
    "project_code",       # A: Project identifier
    "form_code",          # B: CRF form code
    "subject_id",         # C: Subject number
    "initials",           # D: Subject initials
    "subject_status",     # E: Subject status
    "site_code",          # F: Site number
    "site_name",          # G: Site name
    "visit",             # H: Data section / visit
    "instance_order",     # I: Instance sequence
    "block",             # J: Data block
    "block_order",        # K: Block sequence
    "page",              # L: Data page
    "last_modified",      # M: Last modification timestamp
    "row_number",         # N: Row number
]

# ── CDISC common metadata columns (shared across all domains) ────────────────

CDISC_METADATA_VARS = {
    "SITENM", "SITEID", "PSTUDYNM", "PSTUDYID",
    "SUBJID", "SUBJSTA", "ISDEL", "CRFVER",
    "VISIT", "VISTOID", "VISTREP",
    "FORMNM", "FORMOID", "FORMREP", "RECREP",
    "PAGELMBY", "PAGEFSDT", "PAGELMDT",
}

CDISC_TO_INTERNAL = {
    "PSTUDYID": "project_code",
    "SUBJID": "subject_id",
    "SUBJSTA": "subject_status",
    "SITEID": "site_code",
    "SITENM": "site_name",
    "VISIT": "visit",
    "FORMOID": "form_code",
    "PAGELMDT": "last_modified",
}

# ── LB domain subtypes ──────────────────────────────────────────────────────

LB_SUBTYPES = {
    "LB_CHEM", "LB_HEM", "LB_COA", "LB_URI",
    "LB_HBV", "LB_HCV", "LB_HCG", "LB_VIR",
}

# Known oncology-specific domains
ONCOLOGY_DOMAINS = {
    "TL", "TL1", "NTL", "NTL1", "NL", "RS",
    "DLT", "TDT", "TDT1", "TRH", "TSH", "TUT",
    "ECO", "KG", "FP",
}


class EDCFormat(Enum):
    GENERIC = "generic"
    CDISC = "cdisc"


@dataclass
class EDCDataset:
    """Container for a fully parsed EDC export."""

    source_file: str
    project_code: str = ""
    format: EDCFormat = EDCFormat.GENERIC
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    analysis_sheets: dict[str, pd.DataFrame] = field(default_factory=dict)
    toc: pd.DataFrame | None = None
    form_mapping: dict[str, str] = field(default_factory=dict)
    subject_ids: list[str] = field(default_factory=list)
    site_info: dict[str, str] = field(default_factory=dict)
    domain_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def n_subjects(self) -> int:
        all_ids = set()
        for df in self.tables.values():
            if "subject_id" in df.columns:
                all_ids.update(df["subject_id"].dropna().unique())
            elif "SUBJID" in df.columns:
                all_ids.update(df["SUBJID"].dropna().unique())
        return len(all_ids)

    @property
    def n_tables(self) -> int:
        return len(self.tables)

    @property
    def is_oncology(self) -> bool:
        """Detect if this dataset contains oncology-specific domains."""
        return bool(set(self.tables.keys()) & ONCOLOGY_DOMAINS)

    @property
    def lb_subtypes(self) -> list[str]:
        """Return LB domain subtypes present in the dataset."""
        return sorted(set(self.tables.keys()) & LB_SUBTYPES)

    def get_table(self, form_code: str) -> pd.DataFrame | None:
        return self.tables.get(form_code.upper())

    def get_lb(self, subtype: str | None = None) -> pd.DataFrame | None:
        """Get lab data, optionally filtered by subtype (CHEM, HEM, etc.)."""
        if subtype:
            key = f"LB_{subtype.upper()}"
            return self.tables.get(key)
        for key in ["LB", "LB_CHEM", "LB_HEM"]:
            if key in self.tables:
                return self.tables[key]
        return None

    def get_all_lb(self) -> pd.DataFrame:
        """Concatenate all LB subtypes into a single DataFrame."""
        frames = []
        for key in sorted(self.tables.keys()):
            if key.startswith("LB"):
                df = self.tables[key].copy()
                df["_lb_subtype"] = key
                frames.append(df)
        if frames:
            return pd.concat(frames, ignore_index=True, sort=False)
        return pd.DataFrame()

    def list_forms(self) -> list[str]:
        return sorted(self.tables.keys())


class EDCParser:
    """
    Parses EDC export Excel files with automatic format detection.

    Supports:
      - Generic EDC: 14-column Chinese/English header
      - CDISC EDC: Two-row header (Chinese label + CDISC variable code)
    """

    def __init__(self, config_path: str | Path | None = None):
        self.config: dict[str, Any] = {}
        if config_path:
            self._load_config(config_path)

    def _load_config(self, path: str | Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def parse(self, filepath: str | Path) -> EDCDataset:
        """Parse an EDC export Excel file into an EDCDataset."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"EDC file not found: {filepath}")

        xl = pd.ExcelFile(filepath, engine="openpyxl")
        dataset = EDCDataset(source_file=str(filepath))

        detected_format = self._detect_format(xl)
        dataset.format = detected_format

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name, header=None, dtype=str)
            if df.empty:
                continue

            if self._is_toc_sheet(sheet_name, df):
                dataset.toc = df
                self._parse_toc(df, dataset)
            elif self._is_domain_index_sheet(sheet_name, df):
                self._parse_domain_index(df, dataset)
            elif detected_format == EDCFormat.CDISC and self._is_cdisc_data_sheet(df):
                domain = sheet_name.strip().upper()
                parsed = self._standardize_cdisc_sheet(df, domain)
                dataset.tables[domain] = parsed
                if not dataset.project_code and "PSTUDYID" in parsed.columns:
                    vals = parsed["PSTUDYID"].dropna().unique()
                    if len(vals) > 0:
                        dataset.project_code = str(vals[0])
            elif self._is_edc_data_sheet(df):
                form_code = self._extract_form_code(df)
                parsed = self._standardize_edc_sheet(df)
                dataset.tables[form_code] = parsed
                if dataset.project_code == "" and not parsed.empty:
                    dataset.project_code = str(parsed.iloc[0].get("project_code", ""))
            else:
                dataset.analysis_sheets[sheet_name] = df

        dataset.subject_ids = self._collect_subject_ids(dataset)
        dataset.site_info = self._collect_site_info(dataset)
        return dataset

    # ── Format detection ─────────────────────────────────────────────────

    def _detect_format(self, xl: pd.ExcelFile) -> EDCFormat:
        """Auto-detect CDISC vs generic format by sampling sheets."""
        cdisc_score = 0
        generic_score = 0

        sample_sheets = xl.sheet_names[:min(5, len(xl.sheet_names))]
        for sheet_name in sample_sheets:
            df = xl.parse(sheet_name, header=None, dtype=str, nrows=3)
            if df.empty or df.shape[0] < 2:
                continue

            if self._has_cdisc_two_row_header(df):
                cdisc_score += 1
            elif self._is_edc_data_sheet(df):
                generic_score += 1

        return EDCFormat.CDISC if cdisc_score > generic_score else EDCFormat.GENERIC

    def _has_cdisc_two_row_header(self, df: pd.DataFrame) -> bool:
        """Check for the two-row header: Row 0 = Chinese, Row 1 = CDISC codes."""
        if df.shape[0] < 2 or df.shape[1] < 5:
            return False

        row1_values = [str(v).strip().upper() for v in df.iloc[1] if v is not None and str(v).strip()]
        cdisc_hits = sum(1 for v in row1_values if v in CDISC_METADATA_VARS)
        if cdisc_hits >= 3:
            return True

        known_prefixes = {"SUBJID", "SITEID", "VISIT", "FORMOID", "PSTUDYID"}
        if known_prefixes & set(row1_values):
            return True

        return False

    def _is_cdisc_data_sheet(self, df: pd.DataFrame) -> bool:
        """Detect CDISC data sheet (two-row header with domain variables)."""
        return self._has_cdisc_two_row_header(df) and df.shape[0] > 2

    def _is_domain_index_sheet(self, name: str, df: pd.DataFrame) -> bool:
        name_upper = name.strip().upper()
        if name_upper in ("DOMAIN_NAME", "DOMAINS", "DOMAIN_LIST"):
            return True
        if df.shape[1] <= 5 and df.shape[0] < 100:
            row0 = " ".join(str(v) for v in df.iloc[0] if v is not None).upper()
            if "DOMAIN" in row0 and "FORM" in row0:
                return True
        return False

    # ── CDISC-specific parsing ───────────────────────────────────────────

    def _standardize_cdisc_sheet(self, df: pd.DataFrame, domain: str) -> pd.DataFrame:
        """Parse a CDISC two-row header sheet into a clean DataFrame."""
        cn_labels = [str(v).strip() if v is not None else "" for v in df.iloc[0]]
        cdisc_vars = [str(v).strip() if v is not None else "" for v in df.iloc[1]]

        columns = []
        seen = {}
        for i, var in enumerate(cdisc_vars):
            col = var.upper() if var and var.lower() != "nan" else f"_col_{i}"
            if col in seen:
                seen[col] += 1
                col = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
            columns.append(col)

        data_df = df.iloc[2:].reset_index(drop=True)
        data_df.columns = columns

        data_df.attrs["_cn_labels"] = dict(zip(columns, cn_labels))
        data_df.attrs["_domain"] = domain

        subj_col = "SUBJID" if "SUBJID" in data_df.columns else None
        if subj_col:
            data_df[subj_col] = data_df[subj_col].apply(
                lambda x: str(x).strip() if x is not None and str(x).strip().lower() != "nan" else ""
            )

        mapped_cols = {}
        for cdisc_var, internal_name in CDISC_TO_INTERNAL.items():
            if cdisc_var in data_df.columns and internal_name not in data_df.columns:
                mapped_cols[internal_name] = data_df[cdisc_var]

        for name, series in mapped_cols.items():
            data_df[name] = series

        return data_df

    def _parse_domain_index(self, df: pd.DataFrame, dataset: EDCDataset) -> None:
        """Parse DOMAIN_NAME sheet for domain -> form name mapping."""
        for idx in range(df.shape[0]):
            row = df.iloc[idx]
            vals = [str(v).strip() for v in row if v is not None and str(v).strip().lower() != "nan"]
            if len(vals) >= 2:
                code = vals[0].upper()
                name = vals[1]
                if code and len(code) <= 15 and code != "DOMAIN":
                    dataset.form_mapping[code] = name

    # ── Generic EDC parsing (unchanged) ──────────────────────────────────

    def _is_toc_sheet(self, name: str, df: pd.DataFrame) -> bool:
        name_lower = name.lower()
        if name_lower in ("toc", "目录", "contents", "index"):
            return True
        if df.shape[1] == 2 and df.shape[0] < 100:
            return True
        return False

    def _is_edc_data_sheet(self, df: pd.DataFrame) -> bool:
        """Detect if a sheet follows the standard EDC 14-column header pattern."""
        if df.shape[1] < 14:
            return False
        if df.shape[0] < 2:
            return False

        header_row = df.iloc[0]
        header_str = " ".join(str(v) for v in header_row[:14] if v is not None).lower()

        edc_indicators = ["项目编号", "表单编号", "受试者编号", "数据节", "数据页"]
        alt_indicators = ["project", "form", "subject", "visit", "page"]

        cn_match = sum(1 for ind in edc_indicators if ind in header_str)
        en_match = sum(1 for ind in alt_indicators if ind in header_str)

        if cn_match >= 3 or en_match >= 3:
            return True

        first_data = df.iloc[1] if df.shape[0] > 1 else None
        if first_data is not None:
            col_a = str(first_data.iloc[0]) if first_data.iloc[0] else ""
            if re.match(r"[A-Z]-\d+-[A-Z]+-\d+", col_a):
                return True
            col_c = str(first_data.iloc[2]) if first_data.iloc[2] else ""
            if re.match(r"S\d{5}", col_c):
                return True

        return False

    def _extract_form_code(self, df: pd.DataFrame) -> str:
        """Extract the form code from the sheet data."""
        if df.shape[0] > 1 and df.shape[1] > 1:
            val = str(df.iloc[1, 1]).strip() if df.iloc[1, 1] else ""
            if val and len(val) <= 10:
                return val.upper()
        return "UNKNOWN"

    def _standardize_edc_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize an EDC data sheet with consistent column names."""
        header_row = 0
        data_df = df.iloc[header_row + 1:].reset_index(drop=True)

        n_std = min(14, data_df.shape[1])
        col_names = EDC_STANDARD_COLUMNS[:n_std]
        extra_cols = [f"field_{i}" for i in range(data_df.shape[1] - n_std)]
        data_df.columns = col_names + extra_cols

        if "subject_id" in data_df.columns:
            data_df["subject_id"] = data_df["subject_id"].apply(
                lambda x: str(x).strip() if x is not None else ""
            )

        return data_df

    def _parse_toc(self, df: pd.DataFrame, dataset: EDCDataset) -> None:
        """Parse TOC sheet to build form code -> form name mapping."""
        for _, row in df.iterrows():
            code = str(row.iloc[0]).strip() if row.iloc[0] else ""
            name = str(row.iloc[1]).strip() if len(row) > 1 and row.iloc[1] else ""
            if code and name and code.upper() != "NONE":
                dataset.form_mapping[code.upper()] = name

    def _collect_subject_ids(self, dataset: EDCDataset) -> list[str]:
        all_ids = set()
        for df in dataset.tables.values():
            for col in ("subject_id", "SUBJID"):
                if col in df.columns:
                    ids = df[col].dropna().unique()
                    all_ids.update(s for s in ids if s and s != "" and str(s).lower() != "nan")
        return sorted(all_ids)

    def _collect_site_info(self, dataset: EDCDataset) -> dict[str, str]:
        sites = {}
        for df in dataset.tables.values():
            code_col = "site_code" if "site_code" in df.columns else "SITEID" if "SITEID" in df.columns else None
            name_col = "site_name" if "site_name" in df.columns else "SITENM" if "SITENM" in df.columns else None
            if code_col and name_col:
                for _, row in df[[code_col, name_col]].drop_duplicates().iterrows():
                    code = str(row[code_col]).strip()
                    name = str(row[name_col]).strip()
                    if code and code != "" and code.lower() != "nan" and code not in sites:
                        sites[code] = name
        return sites


