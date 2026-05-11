"""
PDF Report Exporter for Medical Monitoring.

Generates a formal, regulatory-grade PDF report with:
- Cover page with project metadata
- Table of Contents
- Executive Summary with KPI indicators
- 22-section structured body
- Priority-classified Action Plan table
- Page headers/footers with version and confidentiality notices
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from ..utils.helpers import Finding


# ─── Color palette (FDA regulatory-inspired) ────────────────────────────

COLOR_PRIMARY = colors.HexColor("#1B3A5C")
COLOR_ACCENT = colors.HexColor("#2E86AB")
COLOR_URGENT = colors.HexColor("#C0392B")
COLOR_HIGH = colors.HexColor("#E67E22")
COLOR_MEDIUM = colors.HexColor("#F39C12")
COLOR_LOW = colors.HexColor("#27AE60")
COLOR_HEADER_BG = colors.HexColor("#EBF5FB")
COLOR_ROW_ALT = colors.HexColor("#F8F9FA")


class PDFReportExporter:
    """Generate formal PDF Medical Monitoring reports."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.styles = getSampleStyleSheet()
        self._register_custom_styles()

    def export(self, report: dict[str, Any], findings: list[Finding] | None = None) -> Path:
        """
        Export a full report dict to PDF.

        Args:
            report: The report dict from MedicalMonitoringReportGenerator.generate_report()
            findings: Optional list of all findings for detailed tables

        Returns:
            Path to the generated PDF file
        """
        doc = self._create_doc()
        story = []

        story.extend(self._build_cover(report.get("metadata", {})))
        story.append(NextPageTemplate("content"))
        story.append(PageBreak())

        story.extend(self._build_executive_summary(report.get("executive_summary", {})))
        story.append(PageBreak())

        story.extend(self._build_action_plan(report.get("action_plan", {})))
        story.append(PageBreak())

        story.extend(self._build_sections(report.get("sections", [])))

        if findings:
            story.append(PageBreak())
            story.extend(self._build_findings_appendix(findings))

        doc.build(story)
        return self.output_path

    # ─── Document setup ──────────────────────────────────────────────

    def _create_doc(self) -> BaseDocTemplate:
        doc = BaseDocTemplate(
            str(self.output_path),
            pagesize=A4,
            leftMargin=2 * cm,
            rightMargin=2 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2 * cm,
        )

        frame_cover = Frame(
            doc.leftMargin, doc.bottomMargin,
            doc.width, doc.height,
            id="cover",
        )
        frame_content = Frame(
            doc.leftMargin, doc.bottomMargin,
            doc.width, doc.height,
            id="content",
        )

        doc.addPageTemplates([
            PageTemplate(id="cover", frames=[frame_cover],
                         onPage=self._cover_page_footer),
            PageTemplate(id="content", frames=[frame_content],
                         onPage=self._content_page_header_footer),
        ])
        return doc

    def _cover_page_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(
            A4[0] / 2, 1.5 * cm,
            "CONFIDENTIAL - For Internal Use Only"
        )
        canvas.restoreState()

    def _content_page_header_footer(self, canvas, doc):
        canvas.saveState()
        # Header
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(COLOR_PRIMARY)
        canvas.drawString(2 * cm, A4[1] - 1.5 * cm, "Medical Monitoring Report")
        canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.5 * cm,
                               f"Generated: {datetime.now().strftime('%Y-%m-%d')}")
        canvas.setStrokeColor(COLOR_PRIMARY)
        canvas.line(2 * cm, A4[1] - 1.7 * cm, A4[0] - 2 * cm, A4[1] - 1.7 * cm)

        # Footer
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0] / 2, 1 * cm, f"Page {doc.page}")
        canvas.drawString(2 * cm, 1 * cm, "CONFIDENTIAL")
        canvas.restoreState()

    # ─── Custom styles ───────────────────────────────────────────────

    def _register_custom_styles(self):
        self.styles.add(ParagraphStyle(
            "CoverTitle",
            parent=self.styles["Title"],
            fontSize=28,
            textColor=COLOR_PRIMARY,
            spaceAfter=30,
            alignment=1,
        ))
        self.styles.add(ParagraphStyle(
            "CoverSubtitle",
            parent=self.styles["Normal"],
            fontSize=14,
            textColor=COLOR_ACCENT,
            spaceAfter=10,
            alignment=1,
        ))
        self.styles.add(ParagraphStyle(
            "SectionHeader",
            parent=self.styles["Heading1"],
            fontSize=14,
            textColor=COLOR_PRIMARY,
            spaceBefore=20,
            spaceAfter=10,
            borderWidth=1,
            borderColor=COLOR_PRIMARY,
            borderPadding=5,
        ))
        self.styles.add(ParagraphStyle(
            "SummaryKPI",
            parent=self.styles["Normal"],
            fontSize=11,
            spaceAfter=6,
        ))
        self.styles.add(ParagraphStyle(
            "FindingDesc",
            parent=self.styles["Normal"],
            fontSize=9,
            leading=12,
        ))

    # ─── Cover page ─────────────────────────────────────────────────

    def _build_cover(self, metadata: dict) -> list:
        elements = []
        elements.append(Spacer(1, 6 * cm))
        elements.append(Paragraph("Medical Monitoring Report", self.styles["CoverTitle"]))
        elements.append(Spacer(1, 1 * cm))

        project = metadata.get("project", "")
        if project:
            elements.append(Paragraph(f"Project: {project}", self.styles["CoverSubtitle"]))

        indication = metadata.get("indication", "")
        if indication:
            elements.append(Paragraph(f"Indication: {indication}", self.styles["CoverSubtitle"]))

        elements.append(Spacer(1, 2 * cm))

        info_lines = [
            f"Subjects: {metadata.get('subjects', 'N/A')}",
            f"Data Cutoff: {metadata.get('data_cutoff', 'N/A')}",
            f"Generated: {metadata.get('generated_at', datetime.now().isoformat())[:10]}",
            f"Framework: v{metadata.get('framework_version', '1.2.0')}",
            f"Regulatory Basis: {metadata.get('regulatory_basis', 'ICH E6(R2)')}",
        ]
        for line in info_lines:
            elements.append(Paragraph(line, self.styles["CoverSubtitle"]))

        return elements

    # ─── Executive summary ───────────────────────────────────────────

    def _build_executive_summary(self, summary: dict) -> list:
        elements = []
        elements.append(Paragraph("Executive Summary", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 5 * mm))

        msg = summary.get("key_message", "")
        if msg:
            elements.append(Paragraph(f"<b>{msg}</b>", self.styles["SummaryKPI"]))
            elements.append(Spacer(1, 5 * mm))

        kpi_data = [
            ["Metric", "Count"],
            ["Total Findings", str(summary.get("total_findings", 0))],
            ["Urgent (P1)", str(summary.get("urgent_count", 0))],
            ["High (P2)", str(summary.get("high_count", 0))],
            ["Medium (P3)", str(summary.get("medium_count", 0))],
        ]

        table = Table(kpi_data, colWidths=[8 * cm, 4 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_ROW_ALT]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)

        top_cats = summary.get("top_finding_categories", [])
        if top_cats:
            elements.append(Spacer(1, 8 * mm))
            elements.append(Paragraph("<b>Top Finding Categories:</b>", self.styles["SummaryKPI"]))
            for cat, count in top_cats[:5]:
                elements.append(Paragraph(
                    f"  - {cat}: {count} findings", self.styles["SummaryKPI"]
                ))

        return elements

    # ─── Action Plan ─────────────────────────────────────────────────

    def _build_action_plan(self, action_plan: dict) -> list:
        elements = []
        elements.append(Paragraph("Action Plan", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 5 * mm))

        for priority, label, color in [
            ("p1_urgent", "P1 - Urgent (24-48h)", COLOR_URGENT),
            ("p2_important", "P2 - Important (This Week)", COLOR_HIGH),
            ("p3_notice", "P3 - Notice (Two Weeks)", COLOR_MEDIUM),
        ]:
            actions = action_plan.get(priority, [])
            if not actions:
                continue

            elements.append(Paragraph(
                f"<b>{label}</b> ({len(actions)} items)",
                self.styles["SummaryKPI"],
            ))

            table_data = [["ID", "Description", "Subject(s)", "Deadline"]]
            for a in actions[:15]:
                desc = a.description[:100] if isinstance(a, dict) else a.description[:100]
                ref_id = a.get("ref_id", "") if isinstance(a, dict) else a.ref_id
                subj = a.get("affected_subjects", "") if isinstance(a, dict) else a.affected_subjects
                dl = a.get("deadline", "") if isinstance(a, dict) else a.deadline
                table_data.append([ref_id, desc, subj, dl])

            table = Table(table_data, colWidths=[2 * cm, 9 * cm, 3 * cm, 2.5 * cm])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_ROW_ALT]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 5 * mm))

        return elements

    # ─── Sections ────────────────────────────────────────────────────

    def _build_sections(self, sections: list) -> list:
        elements = []

        for section in sections:
            num = section.number if hasattr(section, "number") else section.get("number", 0)
            title = section.title if hasattr(section, "title") else section.get("title", "")
            title_en = section.title_en if hasattr(section, "title_en") else section.get("title_en", "")
            findings_count = section.findings_count if hasattr(section, "findings_count") else section.get("findings_count", 0)
            severity_summary = section.severity_summary if hasattr(section, "severity_summary") else section.get("severity_summary", {})
            content = section.content if hasattr(section, "content") else section.get("content")

            header_text = f"Section {num}: {title} ({title_en})"
            elements.append(Paragraph(header_text, self.styles["SectionHeader"]))

            if findings_count > 0:
                sev_str = ", ".join(f"{k}: {v}" for k, v in severity_summary.items())
                elements.append(Paragraph(
                    f"Findings: {findings_count} ({sev_str})",
                    self.styles["SummaryKPI"],
                ))
            else:
                elements.append(Paragraph("No findings.", self.styles["SummaryKPI"]))

            if isinstance(content, pd.DataFrame) and not content.empty:
                elements.append(self._df_to_table(content.head(20)))
            elif isinstance(content, str) and content:
                elements.append(Paragraph(content[:500], self.styles["FindingDesc"]))

            elements.append(Spacer(1, 8 * mm))

        return elements

    # ─── Findings appendix ───────────────────────────────────────────

    def _build_findings_appendix(self, findings: list[Finding]) -> list:
        elements = []
        elements.append(Paragraph("Appendix: All Findings", self.styles["SectionHeader"]))
        elements.append(Spacer(1, 5 * mm))

        table_data = [["Severity", "Category", "Subject", "Description"]]
        for f in findings[:100]:
            table_data.append([
                f.severity.upper(),
                f.category[:25],
                f.subject_id[:10],
                f.description[:80],
            ])

        table = Table(table_data, colWidths=[2 * cm, 4 * cm, 2.5 * cm, 8 * cm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_ROW_ALT]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        elements.append(table)

        if len(findings) > 100:
            elements.append(Paragraph(
                f"... and {len(findings) - 100} more findings (see Excel for full list).",
                self.styles["FindingDesc"],
            ))

        return elements

    # ─── Helpers ─────────────────────────────────────────────────────

    def _df_to_table(self, df: pd.DataFrame) -> Table:
        """Convert a DataFrame to a reportlab Table."""
        cols = list(df.columns)
        data = [cols]
        for _, row in df.iterrows():
            data.append([str(v)[:50] for v in row.values])

        n_cols = len(cols)
        col_width = (A4[0] - 4 * cm) / n_cols
        table = Table(data, colWidths=[col_width] * n_cols)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_HEADER_BG),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_ROW_ALT]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        return table
