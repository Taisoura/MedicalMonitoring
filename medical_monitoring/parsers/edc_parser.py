"""
Module A: EDC Data Parser

Parses multi-sheet EDC export Excel files into a standardized data structure.
Automatically identifies the 14-column header structure common to EDC exports
and separates raw CRF data from analysis/audit sheets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


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


@dataclass
class EDCDataset:
    """Container for a fully parsed EDC export."""

    source_file: str
    project_code: str = ""
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    analysis_sheets: dict[str, pd.DataFrame] = field(default_factory=dict)
    toc: pd.DataFrame | None = None
    form_mapping: dict[str, str] = field(default_factory=dict)
    subject_ids: list[str] = field(default_factory=list)
    site_info: dict[str, str] = field(default_factory=dict)

    @property
    def n_subjects(self) -> int:
        all_ids = set()
        for df in self.tables.values():
            if "subject_id" in df.columns:
                all_ids.update(df["subject_id"].dropna().unique())
        return len(all_ids)

    @property
    def n_tables(self) -> int:
        return len(self.tables)

    def get_table(self, form_code: str) -> pd.DataFrame | None:
        return self.tables.get(form_code.upper())

    def list_forms(self) -> list[str]:
        return sorted(self.tables.keys())


class EDCParser:
    """
    Parses EDC export Excel files with automatic structure detection.

    The parser identifies EDC data sheets (with the standard 14-column header)
    and separates them from analysis/summary sheets.
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

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name, header=None, dtype=str)
            if df.empty:
                continue

            if self._is_toc_sheet(sheet_name, df):
                dataset.toc = df
                self._parse_toc(df, dataset)
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
            if "subject_id" in df.columns:
                ids = df["subject_id"].dropna().unique()
                all_ids.update(s for s in ids if s and s != "")
        return sorted(all_ids)

    def _collect_site_info(self, dataset: EDCDataset) -> dict[str, str]:
        sites = {}
        for df in dataset.tables.values():
            if "site_code" in df.columns and "site_name" in df.columns:
                for _, row in df[["site_code", "site_name"]].drop_duplicates().iterrows():
                    code = str(row["site_code"]).strip()
                    name = str(row["site_name"]).strip()
                    if code and code != "" and code not in sites:
                        sites[code] = name
        return sites


