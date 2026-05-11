"""Shared utility functions for the Medical Monitoring framework."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import pandas as pd


def parse_date_flexible(value: Any) -> datetime | None:
    """Parse dates that may contain 'UK' (unknown) or partial formats."""
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    uk_pattern = re.compile(r"(\d{4})-(\d{2})-[Uu][Kk]")
    m = uk_pattern.match(s)
    if m:
        return None  # date with unknown day
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d", "%d-%b-%Y"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def is_date_incomplete(value: Any) -> bool:
    """Check if a date string contains UK (unknown) portions."""
    if value is None:
        return False
    s = str(value).strip().upper()
    return "UK" in s or "UNK" in s


def normalize_subject_id(sid: Any) -> str:
    """Normalize subject IDs (strip whitespace, uppercase)."""
    if sid is None:
        return ""
    return str(sid).strip().upper()


def safe_float(value: Any) -> float | None:
    """Safely convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(str(value).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


def compute_fold_change(result: float, reference: float, direction: str = "high") -> float | None:
    """Compute fold change vs ULN (high) or LLN (low)."""
    if reference is None or reference == 0:
        return None
    if direction == "high":
        return result / reference
    else:
        return result / reference


class Finding:
    """Represents a single audit finding with severity and context."""

    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    def __init__(
        self,
        category: str,
        subject_id: str,
        description: str,
        severity: str = "medium",
        source_table: str = "",
        row_ref: str = "",
        recommendation: str = "",
        regulatory_basis: str = "",
    ):
        self.category = category
        self.subject_id = subject_id
        self.description = description
        self.severity = severity
        self.source_table = source_table
        self.row_ref = row_ref
        self.recommendation = recommendation
        self.regulatory_basis = regulatory_basis

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "subject_id": self.subject_id,
            "description": self.description,
            "severity": self.severity,
            "source_table": self.source_table,
            "row_ref": self.row_ref,
            "recommendation": self.recommendation,
            "regulatory_basis": self.regulatory_basis,
        }

    def __repr__(self) -> str:
        return f"Finding({self.severity}: {self.subject_id} - {self.description[:60]})"


def findings_to_dataframe(findings: list[Finding]) -> pd.DataFrame:
    """Convert a list of Finding objects to a pandas DataFrame."""
    if not findings:
        return pd.DataFrame(
            columns=["category", "subject_id", "description", "severity",
                     "source_table", "row_ref", "recommendation", "regulatory_basis"]
        )
    return pd.DataFrame([f.to_dict() for f in findings])
