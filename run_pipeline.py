"""
Quick-start script for running the Medical Monitoring pipeline.

Usage:
    python run_pipeline.py <edc_file.xlsx> [--ctcae <ctcae_file.xlsx>] [--output <report.xlsx>]

Example:
    python run_pipeline.py "Y-6-LC-04_FormExcelAllVersion_202604301537_50.xlsx" \
        --ctcae "CTCAE v5.0 Clean.xlsx" \
        --output "medical_monitoring_report.xlsx"
"""

import argparse
import sys
from pathlib import Path

from medical_monitoring.audit.causality import Y6_CAUSALITY_CONFIG, DrugContext, CausalityConfig
from medical_monitoring.analysis.subject_db import STROKE_CONFIG
from medical_monitoring.pipeline import MedicalMonitoringPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Medical Monitoring Framework - Non-Oncology Clinical Trials"
    )
    parser.add_argument("edc_file", help="Path to EDC export Excel file")
    parser.add_argument("--ctcae", help="Path to CTCAE v5.0 Clean.xlsx reference")
    parser.add_argument("--rules", help="Path to audit rules YAML config")
    parser.add_argument("--output", "-o", default="mm_report.xlsx",
                        help="Output report file path")
    parser.add_argument("--indication", default="general",
                        choices=["general", "stroke"],
                        help="Indication-specific configuration")
    parser.add_argument("--viz", action="store_true",
                        help="Generate visualizations after pipeline run")
    parser.add_argument("--viz-dir", default="viz_output",
                        help="Directory for visualization output (default: viz_output)")
    parser.add_argument("--theme", default="fda_regulatory",
                        choices=["fda_regulatory", "academic_conference", "pharma_dashboard"],
                        help="Visualization theme (default: fda_regulatory)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Skip interactive HTML charts, only static images")
    parser.add_argument("--ai", action="store_true",
                        help="Enable AI/LLM semantic enhancement layer")
    parser.add_argument("--ai-config", default=None,
                        help="Path to AI configuration YAML (default: config/ai_config.yaml)")

    output_group = parser.add_argument_group(
        "Output formats",
        "Generate reports in multiple formats. Excel is always generated."
    )
    output_group.add_argument("--pdf", default=None,
                              help="Generate PDF report (provide output path, e.g., report.pdf)")
    output_group.add_argument("--docx", default=None,
                              help="Generate Word report (provide output path, e.g., report.docx)")
    output_group.add_argument("--json", default=None,
                              help="Generate JSON export (provide output path, e.g., results.json)")
    output_group.add_argument("--json-minimal", action="store_true",
                              help="Export minimal JSON (KPIs only, smaller file)")
    output_group.add_argument("--dashboard", action="store_true",
                              help="Launch Streamlit web dashboard after pipeline run")
    output_group.add_argument("--dashboard-port", type=int, default=8501,
                              help="Port for Streamlit dashboard (default: 8501)")

    drug_group = parser.add_argument_group(
        "Drug/IB context",
        "Provide as much or as little as confidentiality allows. "
        "The framework adapts automatically: FULL (IB details) / PARTIAL (drug name only) / BLIND (nothing)."
    )
    drug_group.add_argument("--drug-name", default="",
                            help="Study drug name (e.g., 'Drug X' or generic class)")
    drug_group.add_argument("--drug-class", default="",
                            help="Drug class (e.g., 'antiplatelet', 'PDE3 inhibitor')")
    drug_group.add_argument("--drug-indication", default="",
                            help="Target indication (e.g., 'acute ischemic stroke')")
    drug_group.add_argument("--ib-risks", default="",
                            help="Known risks from IB (comma-separated, e.g., 'bleeding,headache,hepatotoxicity')")
    drug_group.add_argument("--drug-mechanism", default="",
                            help="Brief mechanism of action summary")
    drug_group.add_argument("--protocol-phase", default="",
                            help="Clinical trial phase (e.g., 'Phase II', 'Phase III')")
    drug_group.add_argument("--therapeutic-area", default="",
                            help="Therapeutic area (e.g., 'neurology', 'cardiology')")
    args = parser.parse_args()

    edc_path = Path(args.edc_file)
    if not edc_path.exists():
        print(f"Error: EDC file not found: {edc_path}")
        sys.exit(1)

    rules_path = args.rules
    if rules_path is None:
        default_rules = Path(__file__).parent / "medical_monitoring" / "config" / "audit_rules.yaml"
        if default_rules.exists():
            rules_path = str(default_rules)

    subject_config = STROKE_CONFIG if args.indication == "stroke" else None
    causality_config = Y6_CAUSALITY_CONFIG if args.indication == "stroke" else None

    # Build DrugContext from CLI args (supports FULL/PARTIAL/BLIND)
    drug_context = DrugContext(
        drug_name=args.drug_name,
        drug_class=args.drug_class,
        indication=args.drug_indication,
        mechanism_summary=args.drug_mechanism,
        known_risk_summary=args.ib_risks,
        protocol_phase=args.protocol_phase,
        therapeutic_area=args.therapeutic_area,
    )

    info_level = drug_context.info_level.value.upper()
    drug_display = args.drug_name or "(not provided -- BLIND mode)"

    print(f"{'='*60}")
    print(f"Medical Monitoring Framework v1.2.0")
    print(f"{'='*60}")
    print(f"EDC File: {edc_path.name}")
    print(f"CTCAE Reference: {args.ctcae or 'None'}")
    print(f"Indication: {args.indication}")
    print(f"Drug Context: {drug_display} [{info_level}]")
    print(f"AI Enabled: {args.ai}")
    print(f"Output: {args.output}")
    print(f"{'='*60}")

    if info_level == "BLIND":
        print("\n  [INFO] No drug/IB information provided.")
        print("  Forward causality audit will be skipped.")
        print("  Provide --drug-name for PARTIAL mode, add --ib-risks for FULL mode.")
    elif info_level == "PARTIAL":
        print(f"\n  [INFO] Drug name provided but no IB risk details.")
        print("  Forward causality audit will be skipped.")
        print("  Add --ib-risks to enable IB-driven forward audit.")

    pipeline = MedicalMonitoringPipeline(
        edc_file=edc_path,
        ctcae_file=args.ctcae,
        audit_rules_config=rules_path,
        subject_db_config=subject_config,
        causality_config=causality_config,
        drug_context=drug_context,
        ai_config_path=args.ai_config,
        ai_enabled=args.ai,
    )

    print("\n[Phase 1] Parsing EDC data...")
    results = pipeline.run()

    print(f"\n[Results Summary]")
    if "report" in results:
        stats = results["report"].get("statistics", {})
        summary = results["report"].get("executive_summary", {})
        print(f"  Total findings: {summary.get('total_findings', 0)}")
        print(f"  Urgent (P1): {summary.get('urgent_count', 0)}")
        print(f"  High (P2): {summary.get('high_count', 0)}")
        print(f"  Medium (P3): {summary.get('medium_count', 0)}")
        print(f"  Total actions: {stats.get('total_actions', 0)}")

    # ── Export: Excel (always) ──────────────────────────────────────
    print(f"\n[Export] Excel -> {args.output}")
    pipeline.export(args.output)
    print(f"  Done: {args.output}")

    # ── Export: PDF ──────────────────────────────────────────────────
    if args.pdf:
        print(f"[Export] PDF -> {args.pdf}")
        try:
            pipeline.export_pdf(args.pdf)
            print(f"  Done: {args.pdf}")
        except Exception as e:
            print(f"  Error: {e}")

    # ── Export: Word (.docx) ─────────────────────────────────────────
    if args.docx:
        print(f"[Export] Word -> {args.docx}")
        try:
            pipeline.export_docx(args.docx)
            print(f"  Done: {args.docx}")
        except Exception as e:
            print(f"  Error: {e}")

    # ── Export: JSON ─────────────────────────────────────────────────
    if args.json:
        print(f"[Export] JSON -> {args.json}")
        try:
            pipeline.export_json(args.json, minimal=args.json_minimal)
            print(f"  Done: {args.json}")
        except Exception as e:
            print(f"  Error: {e}")

    # ── Visualizations ───────────────────────────────────────────────
    if args.viz:
        viz_dir = Path(args.viz_dir)
        interactive = not args.no_interactive
        print(f"\n[Visualizations] Generating charts -> {viz_dir}/")
        print(f"  Theme: {args.theme}")
        print(f"  Interactive HTML: {interactive}")
        pipeline.visualize(viz_dir, theme=args.theme, interactive=interactive)
        print(f"  Done: {viz_dir}/")

    # ── Web Dashboard ────────────────────────────────────────────────
    if args.dashboard:
        json_path = args.json or "mm_results.json"
        if not args.json:
            print(f"\n[Dashboard] Exporting JSON for dashboard -> {json_path}")
            pipeline.export_json(json_path)
        print(f"[Dashboard] Launching Streamlit on port {args.dashboard_port}...")
        print(f"  Open: http://localhost:{args.dashboard_port}")
        pipeline.launch_dashboard(json_path, port=args.dashboard_port)


if __name__ == "__main__":
    main()
