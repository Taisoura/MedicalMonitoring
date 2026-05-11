"""
Report Generator: Medical Monitoring Report

Generates a structured medical monitoring report with:
- 22-section template
- Priority-classified action recommendations (P1 urgent / P2 important / P3 notice)
- Automatic summary generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.helpers import Finding, findings_to_dataframe


@dataclass
class ReportSection:
    """A section in the medical monitoring report."""
    number: int
    title: str
    title_en: str
    content: pd.DataFrame | str | dict
    findings_count: int = 0
    severity_summary: dict[str, int] = field(default_factory=dict)


@dataclass
class ActionRecommendation:
    """A prioritized action recommendation."""
    priority: str  # P1, P2, P3
    ref_id: str
    description: str
    affected_subjects: str
    regulatory_basis: str
    deadline: str
    section_ref: int


REPORT_TEMPLATE = [
    (1, "受试者分布与人口统计学", "Subject Disposition & Demographics"),
    (2, "安全性概况", "Safety Profile"),
    (3, "疗效终点", "Efficacy Endpoints"),
    (4, "临床结局", "Clinical Outcomes"),
    (5, "实验室检查概况", "Laboratory Overview"),
    (6, "关键安全性发现", "Key Safety Findings"),
    (7, "CM↔AE 时序交叉核查", "CM-AE Temporal Cross-Check"),
    (8, "MH↔CM 用药合理性", "MH-CM Medication Reasonability"),
    (9, "LB↔AE 漏报检测", "LB-AE Under-reporting Detection"),
    (10, "LB动态链接准确性", "LB Dynamic Link Accuracy"),
    (11, "回归分析与风险因素", "Regression Analysis & Risk Factors"),
    (12, "评分一致性审计", "Scoring Consistency Audit"),
    (13, "退出/中止分析", "Withdrawal/Discontinuation Analysis"),
    (14, "凝血功能与出血关联", "Coagulation & Bleeding Correlation"),
    (15, "血生化安全性信号", "Biochemistry Safety Signals"),
    (16, "药物暴露分析", "Drug Exposure Analysis"),
    (17, "ECG/影像学安全性", "ECG/Imaging Safety"),
    (18, "功能评分(mRS/NIHSS)分析", "Functional Score Analysis"),
    (19, "AE综合交叉审计", "Comprehensive AE Cross-Audit"),
    (20, "临床关联多域聚合", "Multi-Domain Clinical Correlation"),
    (21, "CTCAE分级审核", "CTCAE Grading Audit"),
    (22, "AE数据质量审核", "AE Data Quality Audit"),
]


class MedicalMonitoringReportGenerator:
    """Generates structured Medical Monitoring reports."""

    def __init__(self, project_info: dict[str, str] | None = None):
        self.project_info = project_info or {}
        self.sections: list[ReportSection] = []
        self.actions: list[ActionRecommendation] = []
        self.all_findings: list[Finding] = []

    def generate_report(
        self,
        audit_results: dict[str, Any],
        subject_db: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Generate the complete medical monitoring report from all audit results.

        Args:
            audit_results: Dict with keys for each module's results
            subject_db: Subject-level integrated database
        """
        self.all_findings = self._collect_all_findings(audit_results)
        self._build_sections(audit_results, subject_db)
        self._generate_actions()

        report = {
            "metadata": self._build_metadata(),
            "executive_summary": self._build_executive_summary(),
            "sections": self.sections,
            "action_plan": {
                "p1_urgent": [a for a in self.actions if a.priority == "P1"],
                "p2_important": [a for a in self.actions if a.priority == "P2"],
                "p3_notice": [a for a in self.actions if a.priority == "P3"],
            },
            "statistics": self._compute_report_statistics(),
        }
        return report

    def export_to_excel(self, report: dict, output_path: str | Path) -> None:
        """Export the report to a multi-sheet Excel file."""
        output_path = Path(output_path)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = {
                "Metric": list(report["statistics"].keys()),
                "Value": list(report["statistics"].values()),
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

            # Action Plan
            actions_data = []
            for a in self.actions:
                actions_data.append({
                    "Priority": a.priority,
                    "ID": a.ref_id,
                    "Description": a.description,
                    "Subjects": a.affected_subjects,
                    "Basis": a.regulatory_basis,
                    "Deadline": a.deadline,
                    "Section": a.section_ref,
                })
            pd.DataFrame(actions_data).to_excel(writer, sheet_name="Action Plan", index=False)

            # All Findings
            findings_df = findings_to_dataframe(self.all_findings)
            if not findings_df.empty:
                findings_df.to_excel(writer, sheet_name="All Findings", index=False)

            # Individual section data
            for section in self.sections:
                if isinstance(section.content, pd.DataFrame) and not section.content.empty:
                    sheet_name = f"S{section.number}_{section.title[:20]}"
                    sheet_name = sheet_name.replace("/", "_").replace("↔", "_")[:31]
                    section.content.to_excel(writer, sheet_name=sheet_name, index=False)

    def _build_metadata(self) -> dict:
        return {
            "report_title": "Medical Monitoring Report",
            "project": self.project_info.get("project_code", ""),
            "indication": self.project_info.get("indication", ""),
            "subjects": self.project_info.get("n_subjects", ""),
            "data_cutoff": self.project_info.get("data_cutoff", ""),
            "generated_at": datetime.now().isoformat(),
            "framework_version": "1.0.0",
            "regulatory_basis": "ICH E6(R2), CTCAE v5.0, MedDRA v20.1",
        }

    def _build_executive_summary(self) -> dict:
        urgent = len([f for f in self.all_findings if f.severity == Finding.URGENT])
        high = len([f for f in self.all_findings if f.severity == Finding.HIGH])
        medium = len([f for f in self.all_findings if f.severity == Finding.MEDIUM])

        categories = {}
        for f in self.all_findings:
            categories[f.category] = categories.get(f.category, 0) + 1

        top_categories = sorted(categories.items(), key=lambda x: -x[1])[:5]

        return {
            "total_findings": len(self.all_findings),
            "urgent_count": urgent,
            "high_count": high,
            "medium_count": medium,
            "top_finding_categories": top_categories,
            "key_message": self._generate_key_message(urgent, high),
        }

    def _generate_key_message(self, urgent: int, high: int) -> str:
        if urgent > 5:
            return (
                f"CRITICAL: {urgent} urgent findings require immediate action. "
                f"Safety data integrity may be compromised."
            )
        elif urgent > 0:
            return (
                f"{urgent} urgent and {high} high-priority findings identified. "
                f"Prompt resolution recommended."
            )
        elif high > 10:
            return (
                f"{high} high-priority findings identified across multiple domains. "
                f"Systematic review recommended."
            )
        else:
            return "Routine findings identified. Standard follow-up procedures apply."

    def _build_sections(self, results: dict, subject_db: pd.DataFrame | None) -> None:
        """Build all 22 report sections from audit results."""
        for num, title_cn, title_en in REPORT_TEMPLATE:
            content = self._get_section_content(num, results, subject_db)
            section_findings = [
                f for f in self.all_findings
                if self._finding_belongs_to_section(f, num)
            ]
            severity_map = {}
            for f in section_findings:
                severity_map[f.severity] = severity_map.get(f.severity, 0) + 1

            self.sections.append(ReportSection(
                number=num,
                title=title_cn,
                title_en=title_en,
                content=content,
                findings_count=len(section_findings),
                severity_summary=severity_map,
            ))

    def _get_section_content(
        self, section_num: int, results: dict, subject_db: pd.DataFrame | None
    ) -> pd.DataFrame | str:
        """Map section numbers to their data sources."""
        mapping = {
            7: "cross_domain.cm_ae_temporal",
            8: "cross_domain.mh_cm_reasonability",
            9: "cross_domain.lb_ae_underreporting",
            10: "cross_domain.dynamic_link_accuracy",
            11: "statistics.regression",
            12: "statistics.consistency",
            19: "cross_domain.ae_dd_death_consistency",
            20: "cross_domain.imaging_ecg_ae_consistency",
            21: "ctcae_grading",
            22: "ae_quality",
        }

        key = mapping.get(section_num, "")
        if not key:
            return pd.DataFrame()

        parts = key.split(".")
        data = results
        for part in parts:
            if isinstance(data, dict):
                data = data.get(part, {})
            else:
                return pd.DataFrame()

        if isinstance(data, list):
            return findings_to_dataframe(data) if data and isinstance(data[0], Finding) else pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            return data
        return pd.DataFrame()

    def _finding_belongs_to_section(self, finding: Finding, section: int) -> bool:
        """Determine if a finding belongs to a given section."""
        category_section_map = {
            "CM↔AE Temporal": 7,
            "MH↔CM Reasonability": 8,
            "LB↔AE Under-reporting": 9,
            "Dynamic Link Accuracy": 10,
            "AE↔DD Death Consistency": 19,
            "Imaging/ECG↔AE": 20,
            "CTCAE Grade Discrepancy": 21,
            "G3+ No AE Record": 21,
            "Hy's Law": 21,
            "AE Naming Consistency": 22,
            "AE Date Completeness": 22,
            "AE Data Completeness": 22,
            "AE Outcome Completeness": 22,
            "AE Severity Reasonability": 22,
            "SAE Screening Gap": 22,
            "Drug Action Consistency": 22,
            "Death Event Audit": 22,
            "Forward Causality": 19,
            "Reverse Causality": 19,
            "Causality Consistency": 19,
        }
        return category_section_map.get(finding.category, 0) == section

    def _generate_actions(self) -> None:
        """Generate prioritized action recommendations from findings."""
        urgent_findings = [f for f in self.all_findings if f.severity == Finding.URGENT]
        high_findings = [f for f in self.all_findings if f.severity == Finding.HIGH]
        medium_findings = [f for f in self.all_findings if f.severity == Finding.MEDIUM]

        for i, f in enumerate(urgent_findings, 1):
            self.actions.append(ActionRecommendation(
                priority="P1",
                ref_id=f"P1-{i:03d}",
                description=f.description[:200],
                affected_subjects=f.subject_id,
                regulatory_basis=f.regulatory_basis,
                deadline="24-48h",
                section_ref=self._get_section_for_finding(f),
            ))

        for i, f in enumerate(high_findings[:30], 1):
            self.actions.append(ActionRecommendation(
                priority="P2",
                ref_id=f"P2-{i:03d}",
                description=f.description[:200],
                affected_subjects=f.subject_id,
                regulatory_basis=f.regulatory_basis,
                deadline="This week",
                section_ref=self._get_section_for_finding(f),
            ))

        for i, f in enumerate(medium_findings[:20], 1):
            self.actions.append(ActionRecommendation(
                priority="P3",
                ref_id=f"P3-{i:03d}",
                description=f.description[:200],
                affected_subjects=f.subject_id,
                regulatory_basis=f.regulatory_basis,
                deadline="Two weeks",
                section_ref=self._get_section_for_finding(f),
            ))

    def _get_section_for_finding(self, finding: Finding) -> int:
        category_map = {
            "CM↔AE Temporal": 7,
            "MH↔CM Reasonability": 8,
            "LB↔AE Under-reporting": 9,
            "CTCAE Grade Discrepancy": 21,
            "AE Naming Consistency": 22,
        }
        return category_map.get(finding.category, 0)

    def _collect_all_findings(self, results: dict) -> list[Finding]:
        """Recursively collect all Finding objects from nested results."""
        findings = []
        if isinstance(results, list):
            for item in results:
                if isinstance(item, Finding):
                    findings.append(item)
                elif isinstance(item, dict):
                    findings.extend(self._collect_all_findings(item))
        elif isinstance(results, dict):
            for value in results.values():
                findings.extend(self._collect_all_findings(value))
        return findings

    def _compute_report_statistics(self) -> dict:
        return {
            "total_sections": len(self.sections),
            "total_findings": len(self.all_findings),
            "total_actions": len(self.actions),
            "p1_urgent_actions": len([a for a in self.actions if a.priority == "P1"]),
            "p2_important_actions": len([a for a in self.actions if a.priority == "P2"]),
            "p3_notice_actions": len([a for a in self.actions if a.priority == "P3"]),
            "sections_with_findings": len([s for s in self.sections if s.findings_count > 0]),
        }
