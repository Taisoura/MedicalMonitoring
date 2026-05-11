"""Quick test: verify PDF, Word, JSON exports work with mock data."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
os.environ["PYTHONIOENCODING"] = "utf-8"

from pathlib import Path
from medical_monitoring.utils.helpers import Finding
from medical_monitoring.reporting.report_generator import (
    MedicalMonitoringReportGenerator, ReportSection, ActionRecommendation
)

# Build mock findings
findings = [
    Finding("CM-AE Temporal", "S00001", "CM start after AE resolution", "urgent",
            "AE/CM", "", "Verify CM timing", "ICH E6(R2)"),
    Finding("LB-AE Under-reporting", "S00002", "ALT 5xULN without AE record", "high",
            "LB3/AE", "", "Report as AE", "ICH E6(R2)"),
    Finding("AE Naming Consistency", "S00003", "Multiple variants for Pyrexia", "high",
            "AE", "", "Standardize to MedDRA PT", "MedDRA v20.1"),
    Finding("CTCAE Grade Discrepancy", "S00004", "Graded 1 but Lab shows G3", "medium",
            "AE/LB3", "", "Re-evaluate grade", "CTCAE v5.0"),
    Finding("Causality Consistency", "MULTIPLE", "Same AE inconsistent causality", "medium",
            "AE", "", "Standardize assessment", "WHO-UMC"),
]

# Generate report
gen = MedicalMonitoringReportGenerator({"project_code": "Y-6-LC-04", "n_subjects": "50"})
report = gen.generate_report({"cross_domain": {"cm_ae_temporal": findings[:1]}})
# Inject findings manually for testing
gen.all_findings = findings
gen._generate_actions()
report["action_plan"] = {
    "p1_urgent": [a for a in gen.actions if a.priority == "P1"],
    "p2_important": [a for a in gen.actions if a.priority == "P2"],
    "p3_notice": [a for a in gen.actions if a.priority == "P3"],
}

output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

# Test 1: PDF
print("[1/4] Testing PDF export...")
try:
    from medical_monitoring.reporting.export_pdf import PDFReportExporter
    pdf_path = output_dir / "test_report.pdf"
    exporter = PDFReportExporter(pdf_path)
    exporter.export(report, findings)
    size_kb = pdf_path.stat().st_size / 1024
    print(f"  OK: {pdf_path} ({size_kb:.1f} KB)")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 2: Word
print("[2/4] Testing Word (.docx) export...")
try:
    from medical_monitoring.reporting.export_docx import DocxReportExporter
    docx_path = output_dir / "test_report.docx"
    exporter = DocxReportExporter(docx_path)
    exporter.export(report, findings)
    size_kb = docx_path.stat().st_size / 1024
    print(f"  OK: {docx_path} ({size_kb:.1f} KB)")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 3: JSON (full)
print("[3/4] Testing JSON export...")
try:
    from medical_monitoring.reporting.export_json import JSONReportExporter
    json_path = output_dir / "test_results.json"
    exporter = JSONReportExporter(json_path)
    result = exporter.export(
        pipeline_results={"report": report, "causality": {"info_level": "full"}},
        report=report,
        findings=findings,
    )
    size_kb = json_path.stat().st_size / 1024
    keys = list(result.keys())
    print(f"  OK: {json_path} ({size_kb:.1f} KB)")
    print(f"  Top-level keys: {keys}")
    print(f"  Findings in JSON: {len(result.get('findings', []))}")
except Exception as e:
    print(f"  FAIL: {e}")

# Test 4: JSON minimal
print("[4/4] Testing JSON minimal export...")
try:
    json_min_path = output_dir / "test_results_minimal.json"
    exporter = JSONReportExporter(json_min_path)
    result = exporter.export_minimal({"report": report})
    size_kb = json_min_path.stat().st_size / 1024
    print(f"  OK: {json_min_path} ({size_kb:.1f} KB)")
    print(f"  KPIs: {result.get('kpi', {})}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n" + "="*50)
print("All export tests completed!")
print(f"Output files in: {output_dir.resolve()}")
