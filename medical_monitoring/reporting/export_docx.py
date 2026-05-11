"""
Word (.docx) Report Exporter for Medical Monitoring.

Generates a CRO/Sponsor-ready Word document with:
- Formal cover page
- Table of Contents field
- Executive Summary
- 22-section structure with findings tables
- Action Plan with priority color coding
- Appendices
- Proper styles, headers, footers, page numbering
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn

from ..utils.helpers import Finding


# ─── Color constants ─────────────────────────────────────────────────

RGB_PRIMARY = RGBColor(0x1B, 0x3A, 0x5C)
RGB_ACCENT = RGBColor(0x2E, 0x86, 0xAB)
RGB_URGENT = RGBColor(0xC0, 0x39, 0x2B)
RGB_HIGH = RGBColor(0xE6, 0x7E, 0x22)
RGB_MEDIUM = RGBColor(0xF3, 0x9C, 0x12)
RGB_LOW = RGBColor(0x27, 0xAE, 0x60)


class DocxReportExporter:
    """Generate formal Word Medical Monitoring reports."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.doc = Document()
        self._setup_styles()

    def export(self, report: dict[str, Any], findings: list[Finding] | None = None) -> Path:
        """
        Export a full report dict to Word.

        Args:
            report: The report dict from MedicalMonitoringReportGenerator.generate_report()
            findings: Optional list of all findings for appendix tables

        Returns:
            Path to the generated .docx file
        """
        self._build_cover(report.get("metadata", {}))
        self._add_toc()
        self._build_executive_summary(report.get("executive_summary", {}))
        self._build_action_plan(report.get("action_plan", {}))
        self._build_sections(report.get("sections", []))

        if findings:
            self._build_findings_appendix(findings)

        self.doc.save(str(self.output_path))
        return self.output_path

    # ─── Document styles ─────────────────────────────────────────────

    def _setup_styles(self):
        """Configure document default styles."""
        style = self.doc.styles["Normal"]
        font = style.font
        font.name = "Calibri"
        font.size = Pt(10)

        sections = self.doc.sections
        for section in sections:
            section.top_margin = Cm(2.5)
            section.bottom_margin = Cm(2)
            section.left_margin = Cm(2.5)
            section.right_margin = Cm(2)

    # ─── Cover page ─────────────────────────────────────────────────

    def _build_cover(self, metadata: dict):
        for _ in range(6):
            self.doc.add_paragraph()

        title = self.doc.add_paragraph()
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = title.add_run("Medical Monitoring Report")
        run.font.size = Pt(28)
        run.font.color.rgb = RGB_PRIMARY
        run.bold = True

        self.doc.add_paragraph()

        project = metadata.get("project", "")
        if project:
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(f"Project: {project}")
            run.font.size = Pt(16)
            run.font.color.rgb = RGB_ACCENT

        indication = metadata.get("indication", "")
        if indication:
            p = self.doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(f"Indication: {indication}")
            run.font.size = Pt(14)
            run.font.color.rgb = RGB_ACCENT

        for _ in range(4):
            self.doc.add_paragraph()

        info_lines = [
            ("Subjects", metadata.get("subjects", "N/A")),
            ("Data Cutoff", metadata.get("data_cutoff", "N/A")),
            ("Generated", metadata.get("generated_at", datetime.now().isoformat())[:10]),
            ("Framework Version", metadata.get("framework_version", "1.2.0")),
            ("Regulatory Basis", metadata.get("regulatory_basis", "ICH E6(R2)")),
        ]

        info_table = self.doc.add_table(rows=len(info_lines), cols=2)
        info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        for i, (label, value) in enumerate(info_lines):
            info_table.cell(i, 0).text = label
            info_table.cell(i, 1).text = str(value)
            for cell in info_table.rows[i].cells:
                for paragraph in cell.paragraphs:
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

        self.doc.add_paragraph()
        conf = self.doc.add_paragraph()
        conf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = conf.add_run("CONFIDENTIAL - For Internal Use Only")
        run.font.size = Pt(9)
        run.italic = True

        self.doc.add_page_break()

    # ─── Table of Contents ───────────────────────────────────────────

    def _add_toc(self):
        self.doc.add_heading("Table of Contents", level=1)
        p = self.doc.add_paragraph()
        run = p.add_run()
        fld_char_begin = run._r.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "begin"})
        run._r.append(fld_char_begin)

        run2 = p.add_run()
        instr_text = run2._r.makeelement(qn("w:instrText"), {})
        instr_text.text = ' TOC \\o "1-3" \\h \\z \\u '
        run2._r.append(instr_text)

        run3 = p.add_run()
        fld_char_end = run3._r.makeelement(qn("w:fldChar"), {qn("w:fldCharType"): "end"})
        run3._r.append(fld_char_end)

        note = self.doc.add_paragraph()
        run = note.add_run("(Update this field in Word: right-click > Update Field)")
        run.font.size = Pt(8)
        run.italic = True

        self.doc.add_page_break()

    # ─── Executive summary ───────────────────────────────────────────

    def _build_executive_summary(self, summary: dict):
        self.doc.add_heading("Executive Summary", level=1)

        msg = summary.get("key_message", "")
        if msg:
            p = self.doc.add_paragraph()
            run = p.add_run(msg)
            run.bold = True
            run.font.size = Pt(11)

        self.doc.add_paragraph()

        kpi_table = self.doc.add_table(rows=5, cols=2)
        kpi_table.style = "Light Grid Accent 1"
        kpi_data = [
            ("Total Findings", summary.get("total_findings", 0)),
            ("Urgent (P1)", summary.get("urgent_count", 0)),
            ("High (P2)", summary.get("high_count", 0)),
            ("Medium (P3)", summary.get("medium_count", 0)),
            ("Key Categories", len(summary.get("top_finding_categories", []))),
        ]
        for i, (label, value) in enumerate(kpi_data):
            kpi_table.cell(i, 0).text = label
            kpi_table.cell(i, 1).text = str(value)

        self.doc.add_paragraph()

        top_cats = summary.get("top_finding_categories", [])
        if top_cats:
            self.doc.add_heading("Top Finding Categories", level=2)
            for cat, count in top_cats[:5]:
                self.doc.add_paragraph(f"{cat}: {count} findings", style="List Bullet")

        self.doc.add_page_break()

    # ─── Action Plan ─────────────────────────────────────────────────

    def _build_action_plan(self, action_plan: dict):
        self.doc.add_heading("Action Plan", level=1)

        for priority_key, label, rgb in [
            ("p1_urgent", "P1 - Urgent (24-48h)", RGB_URGENT),
            ("p2_important", "P2 - Important (This Week)", RGB_HIGH),
            ("p3_notice", "P3 - Notice (Two Weeks)", RGB_MEDIUM),
        ]:
            actions = action_plan.get(priority_key, [])
            if not actions:
                continue

            h = self.doc.add_heading(f"{label} ({len(actions)} items)", level=2)
            for run in h.runs:
                run.font.color.rgb = rgb

            table = self.doc.add_table(rows=1, cols=4)
            table.style = "Table Grid"
            hdr = table.rows[0].cells
            hdr[0].text = "ID"
            hdr[1].text = "Description"
            hdr[2].text = "Subject(s)"
            hdr[3].text = "Deadline"

            for cell in hdr:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True

            for a in actions[:20]:
                row = table.add_row().cells
                if hasattr(a, "ref_id"):
                    row[0].text = a.ref_id
                    row[1].text = a.description[:120]
                    row[2].text = a.affected_subjects
                    row[3].text = a.deadline
                else:
                    row[0].text = a.get("ref_id", "")
                    row[1].text = a.get("description", "")[:120]
                    row[2].text = a.get("affected_subjects", "")
                    row[3].text = a.get("deadline", "")

            self.doc.add_paragraph()

        self.doc.add_page_break()

    # ─── Sections ────────────────────────────────────────────────────

    def _build_sections(self, sections: list):
        self.doc.add_heading("Detailed Findings by Section", level=1)

        for section in sections:
            num = section.number if hasattr(section, "number") else section.get("number", 0)
            title = section.title if hasattr(section, "title") else section.get("title", "")
            title_en = section.title_en if hasattr(section, "title_en") else section.get("title_en", "")
            findings_count = section.findings_count if hasattr(section, "findings_count") else section.get("findings_count", 0)
            severity_summary = section.severity_summary if hasattr(section, "severity_summary") else section.get("severity_summary", {})
            content = section.content if hasattr(section, "content") else section.get("content")

            self.doc.add_heading(f"Section {num}: {title} ({title_en})", level=2)

            if findings_count > 0:
                sev_parts = [f"{k}: {v}" for k, v in severity_summary.items()]
                self.doc.add_paragraph(
                    f"Findings: {findings_count} ({', '.join(sev_parts)})"
                )
            else:
                self.doc.add_paragraph("No findings in this section.")

            if isinstance(content, pd.DataFrame) and not content.empty:
                self._add_dataframe_table(content.head(15))
            elif isinstance(content, str) and content:
                self.doc.add_paragraph(content[:500])

            self.doc.add_paragraph()

    # ─── Findings appendix ───────────────────────────────────────────

    def _build_findings_appendix(self, findings: list[Finding]):
        self.doc.add_page_break()
        self.doc.add_heading("Appendix: All Findings", level=1)

        table = self.doc.add_table(rows=1, cols=5)
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        headers = ["Severity", "Category", "Subject", "Description", "Recommendation"]
        for i, h in enumerate(headers):
            hdr[i].text = h
            for paragraph in hdr[i].paragraphs:
                for run in paragraph.runs:
                    run.bold = True
                    run.font.size = Pt(8)

        for f in findings[:80]:
            row = table.add_row().cells
            row[0].text = f.severity.upper()
            row[1].text = f.category[:30]
            row[2].text = f.subject_id[:12]
            row[3].text = f.description[:80]
            row[4].text = f.recommendation[:60]

            for cell in row:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(7)

        if len(findings) > 80:
            self.doc.add_paragraph(
                f"... and {len(findings) - 80} more findings (see Excel export for complete list)."
            )

    # ─── Helpers ─────────────────────────────────────────────────────

    def _add_dataframe_table(self, df: pd.DataFrame):
        """Add a DataFrame as a Word table."""
        cols = list(df.columns)
        table = self.doc.add_table(rows=1, cols=len(cols))
        table.style = "Light Shading Accent 1"

        hdr = table.rows[0].cells
        for i, col in enumerate(cols):
            hdr[i].text = str(col)[:20]
            for paragraph in hdr[i].paragraphs:
                for run in paragraph.runs:
                    run.bold = True
                    run.font.size = Pt(8)

        for _, row_data in df.head(15).iterrows():
            row = table.add_row().cells
            for i, val in enumerate(row_data.values):
                row[i].text = str(val)[:50]
                for paragraph in row[i].paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(7)
