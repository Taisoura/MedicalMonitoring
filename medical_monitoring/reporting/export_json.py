"""
JSON/API Structured Export for Medical Monitoring.

Serializes the entire pipeline result into a portable JSON structure
suitable for:
- REST API responses
- Cross-system integration (CTMS, EDC, signal management)
- Archival storage
- Frontend dashboard consumption
- Downstream analytics pipelines

Output adheres to a stable schema with versioning for backward compatibility.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.helpers import Finding


SCHEMA_VERSION = "2.0.0"


class JSONReportExporter:
    """Serialize Medical Monitoring results to structured JSON."""

    def __init__(self, output_path: str | Path | None = None, pretty: bool = True):
        self.output_path = Path(output_path) if output_path else None
        self.pretty = pretty

    def export(
        self,
        pipeline_results: dict[str, Any],
        report: dict[str, Any] | None = None,
        findings: list[Finding] | None = None,
        subject_db: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Export pipeline results to JSON structure.

        Args:
            pipeline_results: Raw dict from MedicalMonitoringPipeline.run()
            report: Processed report dict (if already generated)
            findings: All findings
            subject_db: Subject-level database

        Returns:
            The complete JSON-serializable dict (also written to file if output_path set)
        """
        output = {
            "$schema": "medical_monitoring_report",
            "$version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "metadata": self._extract_metadata(report or pipeline_results),
            "executive_summary": self._extract_summary(report),
            "action_plan": self._extract_actions(report),
            "modules": self._extract_modules(pipeline_results),
            "findings": self._serialize_findings(findings),
            "statistics": self._extract_statistics(pipeline_results, report),
            "subject_summary": self._serialize_subject_db(subject_db),
            "ai_results": self._extract_ai_results(pipeline_results),
        }

        if self.output_path:
            self._write_json(output)

        return output

    def export_minimal(self, pipeline_results: dict[str, Any]) -> dict[str, Any]:
        """
        Export a lightweight summary (for API endpoints with size limits).

        Returns only KPIs, action counts, and top findings.
        """
        report = pipeline_results.get("report", {})
        summary = report.get("executive_summary", {})
        stats = report.get("statistics", {})

        output = {
            "$schema": "medical_monitoring_report_minimal",
            "$version": SCHEMA_VERSION,
            "generated_at": datetime.now().isoformat(),
            "kpi": {
                "total_findings": summary.get("total_findings", 0),
                "urgent_count": summary.get("urgent_count", 0),
                "high_count": summary.get("high_count", 0),
                "medium_count": summary.get("medium_count", 0),
                "total_actions": stats.get("total_actions", 0),
                "sections_with_findings": stats.get("sections_with_findings", 0),
            },
            "key_message": summary.get("key_message", ""),
            "top_categories": summary.get("top_finding_categories", []),
            "action_counts": {
                "p1_urgent": stats.get("p1_urgent_actions", 0),
                "p2_important": stats.get("p2_important_actions", 0),
                "p3_notice": stats.get("p3_notice_actions", 0),
            },
            "info_level": pipeline_results.get("causality", {}).get("info_level", "unknown"),
        }

        if self.output_path:
            self._write_json(output)

        return output

    # ─── Extraction helpers ──────────────────────────────────────────

    def _extract_metadata(self, source: dict) -> dict:
        if "metadata" in source:
            return source["metadata"]
        report = source.get("report", {})
        return report.get("metadata", {})

    def _extract_summary(self, report: dict | None) -> dict:
        if not report:
            return {}
        summary = report.get("executive_summary", {})
        return {
            "total_findings": summary.get("total_findings", 0),
            "urgent_count": summary.get("urgent_count", 0),
            "high_count": summary.get("high_count", 0),
            "medium_count": summary.get("medium_count", 0),
            "top_finding_categories": summary.get("top_finding_categories", []),
            "key_message": summary.get("key_message", ""),
        }

    def _extract_actions(self, report: dict | None) -> dict:
        if not report:
            return {"p1_urgent": [], "p2_important": [], "p3_notice": []}

        action_plan = report.get("action_plan", {})
        result = {}
        for key in ("p1_urgent", "p2_important", "p3_notice"):
            actions = action_plan.get(key, [])
            result[key] = [
                {
                    "ref_id": a.ref_id if hasattr(a, "ref_id") else a.get("ref_id", ""),
                    "description": a.description if hasattr(a, "description") else a.get("description", ""),
                    "affected_subjects": a.affected_subjects if hasattr(a, "affected_subjects") else a.get("affected_subjects", ""),
                    "regulatory_basis": a.regulatory_basis if hasattr(a, "regulatory_basis") else a.get("regulatory_basis", ""),
                    "deadline": a.deadline if hasattr(a, "deadline") else a.get("deadline", ""),
                    "section_ref": a.section_ref if hasattr(a, "section_ref") else a.get("section_ref", 0),
                }
                for a in actions
            ]
        return result

    def _extract_modules(self, results: dict) -> dict:
        """Extract per-module summaries."""
        modules = {}

        # Cross-domain
        cross = results.get("cross_domain", {})
        if cross:
            modules["cross_domain"] = {
                "rules_executed": list(cross.keys()),
                "total_findings": sum(
                    len(v) for v in cross.values() if isinstance(v, list)
                ),
            }

        # CTCAE
        ctcae = results.get("ctcae_grading", {})
        if ctcae:
            summary = ctcae.get("summary", {})
            modules["ctcae_grading"] = {
                "total_graded": summary.get("total_graded", 0),
                "discrepancies": summary.get("discrepancies", 0),
                "hys_law_cases": summary.get("hys_law_cases", 0),
                "grade_distribution": summary.get("grade_distribution", {}),
            }

        # Causality
        causality = results.get("causality", {})
        if causality:
            modules["causality"] = {
                "info_level": causality.get("info_level", "unknown"),
                "forward_issues": len(causality.get("forward_findings", [])),
                "reverse_downgrades": len(causality.get("reverse_findings", [])),
                "consistency_issues": len(causality.get("consistency_findings", [])),
                "ai_assessments": len(causality.get("ai_causality_assessments", [])),
                "summary": causality.get("summary", {}),
            }

        # AE Quality
        aeq = results.get("ae_quality", {})
        if aeq:
            modules["ae_quality"] = {
                "dimensions_audited": 10,
                "naming_variants": len(aeq.get("1_naming_variants", [])),
                "action_items": len(aeq.get("10_action_plan", [])),
            }

        # Statistics
        stats = results.get("statistics", {})
        if stats:
            modules["statistics"] = {
                "recommended_methods": stats.get("recommended_methods", []),
                "or_analyses": len(stats.get("mortality_or", [])),
            }

        return modules

    def _serialize_findings(self, findings: list[Finding] | None) -> list[dict]:
        if not findings:
            return []
        return [f.to_dict() for f in findings]

    def _extract_statistics(self, results: dict, report: dict | None) -> dict:
        stats = {}
        if report:
            stats["report_statistics"] = report.get("statistics", {})
        pipeline_stats = results.get("statistics", {})
        if pipeline_stats:
            or_data = pipeline_stats.get("mortality_or", [])
            if or_data:
                stats["or_analyses"] = or_data[:20]
            stats["recommended_methods"] = pipeline_stats.get("recommended_methods", [])
        return stats

    def _serialize_subject_db(self, subject_db: pd.DataFrame | None) -> dict:
        if subject_db is None or subject_db.empty:
            return {"n_subjects": 0, "columns": [], "sample": []}

        return {
            "n_subjects": len(subject_db),
            "columns": list(subject_db.columns),
            "dtypes": {col: str(dt) for col, dt in subject_db.dtypes.items()},
            "sample": subject_db.head(5).to_dict(orient="records"),
            "summary_stats": subject_db.describe().to_dict() if not subject_db.empty else {},
        }

    def _extract_ai_results(self, results: dict) -> dict:
        ai = results.get("ai_preprocessing", {})
        if not ai:
            return {"enabled": False}

        return {
            "enabled": True,
            "field_annotations": bool(ai.get("annotations")),
            "ae_cleaning": ai.get("ae_cleaning", {}),
            "ae_normalization": ai.get("ae_normalization", {}),
        }

    # ─── File I/O ────────────────────────────────────────────────────

    def _write_json(self, data: dict):
        """Write JSON with proper encoding and formatting."""
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(
                data, f,
                ensure_ascii=False,
                indent=2 if self.pretty else None,
                default=self._json_default,
            )

    @staticmethod
    def _json_default(obj):
        """Handle non-serializable types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        return str(obj)
