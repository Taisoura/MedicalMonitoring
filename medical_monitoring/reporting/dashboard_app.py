"""
Streamlit Web Dashboard for Medical Monitoring.

Full-featured interactive dashboard with:
- File upload (EDC Excel + optional CTCAE)
- Configuration sidebar (indication, drug context, AI toggle)
- Live pipeline execution with progress tracking
- KPI summary cards
- Findings drill-down by severity/category/subject
- Action Plan tracker with status
- Module-level detail views
- Oncology-specific views (RECIST, waterfall, DLT summary)
- AI enhancement results
- Multi-format export downloads

Launch:
    streamlit run medical_monitoring/reporting/dashboard_app.py

Or programmatically:
    from medical_monitoring.reporting.dashboard_app import launch_dashboard
    launch_dashboard(port=8501)
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

try:
    import streamlit as st
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False


def launch_dashboard(results_path: str | Path | None = None, port: int = 8501):
    """Launch the Streamlit dashboard programmatically."""
    import subprocess
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).resolve()),
        "--server.port", str(port),
    ]
    if results_path:
        cmd.extend(["--", "--data", str(results_path)])
    subprocess.Popen(cmd)


def load_results(path: str | Path) -> dict:
    """Load JSON results from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_pipeline_thread(
    edc_path: str,
    ctcae_path: str | None,
    indication: str,
    ai_enabled: bool,
    drug_context: dict | None,
    output_json: str,
    status_holder: dict,
):
    """Run pipeline in a background thread, updating status_holder."""
    try:
        status_holder["status"] = "running"
        status_holder["progress"] = 0.1
        status_holder["message"] = "Parsing EDC data..."

        from ..pipeline import MedicalMonitoringPipeline
        from ..audit.causality import DrugContext

        drug_ctx = None
        if drug_context and drug_context.get("drug_name"):
            drug_ctx = DrugContext(
                drug_name=drug_context.get("drug_name", ""),
                drug_class=drug_context.get("drug_class", ""),
                indication=drug_context.get("indication", ""),
                mechanism=drug_context.get("mechanism", ""),
                known_risks=drug_context.get("known_risks", []),
                protocol_phase=drug_context.get("protocol_phase", ""),
            )

        pipeline = MedicalMonitoringPipeline(
            edc_file=edc_path,
            ctcae_file=ctcae_path,
            indication=indication,
            ai_enabled=ai_enabled,
            drug_context=drug_ctx,
        )

        status_holder["progress"] = 0.3
        status_holder["message"] = "Running audit modules..."

        results = pipeline.run()

        status_holder["progress"] = 0.7
        status_holder["message"] = "Generating exports..."

        pipeline.export_json(output_json)

        status_holder["progress"] = 1.0
        status_holder["status"] = "completed"
        status_holder["message"] = "Pipeline completed"

    except Exception as e:
        status_holder["status"] = "failed"
        status_holder["message"] = f"Error: {str(e)}"
        status_holder["error"] = str(e)


# ═══════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════

def main():
    if not HAS_STREAMLIT:
        print("Error: streamlit not installed. Run: pip install streamlit plotly")
        sys.exit(1)

    st.set_page_config(
        page_title="Medical Monitoring Dashboard",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "pipeline_status" not in st.session_state:
        st.session_state.pipeline_status = None
    if "results_data" not in st.session_state:
        st.session_state.results_data = None
    if "results_path" not in st.session_state:
        st.session_state.results_path = None

    # ── Sidebar: Configuration ────────────────────────────────────────
    st.sidebar.title("Medical Monitoring")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Mode",
        ["Upload & Run", "View Results"],
        index=0,
    )

    if mode == "Upload & Run":
        _sidebar_upload_and_config()
    else:
        _sidebar_load_results()

    # ── Main Content ──────────────────────────────────────────────────
    data = st.session_state.results_data
    if data is None:
        _render_welcome()
        return

    _render_dashboard(data)


def _sidebar_upload_and_config():
    """Sidebar for file upload and pipeline configuration."""
    st.sidebar.subheader("1. Upload EDC File")
    edc_file = st.sidebar.file_uploader(
        "EDC Excel (.xlsx)",
        type=["xlsx", "xls"],
        key="edc_upload",
    )

    ctcae_file = st.sidebar.file_uploader(
        "CTCAE Reference (optional)",
        type=["xlsx"],
        key="ctcae_upload",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Configuration")

    indication = st.sidebar.selectbox(
        "Indication",
        ["general", "oncology", "stroke", "cardiovascular"],
        index=0,
    )

    ai_enabled = st.sidebar.checkbox("Enable AI Enhancement", value=False)

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Drug Context (optional)")

    with st.sidebar.expander("Drug / IB Information"):
        drug_name = st.text_input("Drug Name")
        drug_class = st.text_input("Drug Class")
        drug_mechanism = st.text_input("Mechanism of Action")
        protocol_phase = st.text_input("Protocol Phase (e.g., Phase I)")
        known_risks_text = st.text_area("Known Risks (one per line)")

    st.sidebar.markdown("---")

    if st.sidebar.button("Run Pipeline", type="primary", disabled=edc_file is None):
        if edc_file is None:
            st.sidebar.error("Please upload an EDC file first")
            return

        tmp_dir = Path(tempfile.mkdtemp(prefix="medmon_"))
        edc_path = tmp_dir / edc_file.name
        with open(edc_path, "wb") as f:
            f.write(edc_file.getvalue())

        ctcae_path = None
        if ctcae_file:
            ctcae_path = str(tmp_dir / ctcae_file.name)
            with open(ctcae_path, "wb") as f:
                f.write(ctcae_file.getvalue())

        output_json = str(tmp_dir / "results.json")
        st.session_state.results_path = output_json

        drug_context = {
            "drug_name": drug_name,
            "drug_class": drug_class,
            "mechanism": drug_mechanism,
            "protocol_phase": protocol_phase,
            "known_risks": [r.strip() for r in known_risks_text.split("\n") if r.strip()],
        }

        status_holder = {"status": "starting", "progress": 0, "message": "", "error": None}
        st.session_state.pipeline_status = status_holder

        t = threading.Thread(
            target=_run_pipeline_thread,
            args=(str(edc_path), ctcae_path, indication, ai_enabled,
                  drug_context, output_json, status_holder),
            daemon=True,
        )
        t.start()
        st.rerun()

    status = st.session_state.pipeline_status
    if status and status.get("status") in ("starting", "running"):
        st.sidebar.progress(status.get("progress", 0))
        st.sidebar.info(status.get("message", "Running..."))
        time.sleep(1)
        st.rerun()

    elif status and status.get("status") == "completed":
        st.sidebar.success("Pipeline completed!")
        rpath = st.session_state.results_path
        if rpath and Path(rpath).exists():
            st.session_state.results_data = load_results(rpath)
            st.session_state.pipeline_status = None
            st.rerun()

    elif status and status.get("status") == "failed":
        st.sidebar.error(f"Pipeline failed: {status.get('message', '')}")


def _sidebar_load_results():
    """Sidebar for loading existing results."""
    st.sidebar.subheader("Load Results")

    data_path = st.sidebar.text_input(
        "Results JSON path",
        value="mm_results.json",
    )

    uploaded_json = st.sidebar.file_uploader(
        "Or upload JSON file",
        type=["json"],
        key="json_upload",
    )

    if st.sidebar.button("Load", type="primary"):
        try:
            if uploaded_json:
                st.session_state.results_data = json.loads(uploaded_json.getvalue())
                st.rerun()
            elif Path(data_path).exists():
                st.session_state.results_data = load_results(data_path)
                st.rerun()
            else:
                st.sidebar.error(f"File not found: {data_path}")
        except Exception as e:
            st.sidebar.error(f"Error loading: {e}")


def _render_welcome():
    """Welcome page when no data is loaded."""
    st.title("Medical Monitoring Dashboard")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Getting Started

        **Option A: Upload & Run**
        1. Upload your EDC Excel file in the sidebar
        2. Configure indication and drug context
        3. Click "Run Pipeline"
        4. View results in the dashboard

        **Option B: View Existing Results**
        1. Switch to "View Results" mode
        2. Load a previously generated JSON file
        """)

    with col2:
        st.markdown("""
        ### Supported Features

        - Non-oncology medical monitoring (stroke, cardiovascular)
        - Oncology modules (RECIST 1.1, dose escalation, hepatotoxicity)
        - AI/LLM enhancement (Qwen + GPT)
        - Cross-domain auditing (CM-AE, LB-AE, MH-CM)
        - CTCAE v5.0 grading
        - Multi-format export (PDF, Word, Excel, JSON)
        """)

    st.markdown("---")
    st.markdown("""
    **CLI Alternative:**
    ```bash
    python run_pipeline.py <edc_file.xlsx> --json results.json --dashboard
    ```
    """)


def _render_dashboard(data: dict):
    """Render the full dashboard from loaded results data."""
    st.title("Medical Monitoring Dashboard")

    meta = data.get("metadata", {})
    col1, col2, col3, col4 = st.columns(4)
    col1.caption(f"Project: {meta.get('project', 'N/A')}")
    col2.caption(f"Subjects: {meta.get('n_subjects', '?')}")
    col3.caption(f"Generated: {data.get('generated_at', '')[:10] if data.get('generated_at') else 'N/A'}")
    is_onc = data.get("metadata", {}).get("is_oncology", False)
    col4.caption(f"Indication: {'Oncology' if is_onc else meta.get('indication', 'General')}")

    st.markdown("---")

    # ── KPI Cards ────────────────────────────────────────────────────
    summary = data.get("executive_summary", {})
    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Total Findings", summary.get("total_findings", 0))
    kpi_cols[1].metric("Urgent (P1)", summary.get("urgent_count", 0),
                       delta="Action!" if summary.get("urgent_count", 0) > 0 else None,
                       delta_color="inverse")
    kpi_cols[2].metric("High (P2)", summary.get("high_count", 0))
    kpi_cols[3].metric("Medium (P3)", summary.get("medium_count", 0))

    action_plan = data.get("action_plan", {})
    total_actions = (len(action_plan.get("p1_urgent", []))
                     + len(action_plan.get("p2_important", []))
                     + len(action_plan.get("p3_notice", [])))
    kpi_cols[4].metric("Total Actions", total_actions)
    kpi_cols[5].metric("Subjects", meta.get("n_subjects", "?"))

    msg = summary.get("key_message", "")
    if msg:
        if "CRITICAL" in msg:
            st.error(msg)
        elif "urgent" in msg.lower():
            st.warning(msg)
        else:
            st.info(msg)

    st.markdown("---")

    # ── Tabs ─────────────────────────────────────────────────────────
    tab_names = ["Overview", "Action Plan", "Findings", "Modules", "AI Results"]
    if is_onc:
        tab_names.insert(3, "Oncology")
    tabs = st.tabs(tab_names)

    idx = 0
    with tabs[idx]:
        _render_overview(data, summary)
    idx += 1

    with tabs[idx]:
        _render_action_plan(action_plan)
    idx += 1

    with tabs[idx]:
        _render_findings(data.get("findings", []))
    idx += 1

    if is_onc:
        with tabs[idx]:
            _render_oncology(data)
        idx += 1

    with tabs[idx]:
        _render_modules(data.get("modules", {}))
    idx += 1

    with tabs[idx]:
        _render_ai(data.get("ai_results", {}))

    # ── Export Section ────────────────────────────────────────────────
    st.markdown("---")
    _render_export_section(data)


# ═══════════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ═══════════════════════════════════════════════════════════════════════

def _render_overview(data: dict, summary: dict):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Finding Categories")
        cats = summary.get("top_finding_categories", [])
        if cats:
            df_cats = pd.DataFrame(cats, columns=["Category", "Count"])
            fig = px.bar(df_cats, x="Count", y="Category", orientation="h",
                         color="Count", color_continuous_scale="Reds")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No findings to display.")

    with col2:
        st.subheader("Severity Distribution")
        sev_data = {
            "Urgent": summary.get("urgent_count", 0),
            "High": summary.get("high_count", 0),
            "Medium": summary.get("medium_count", 0),
        }
        if any(sev_data.values()):
            fig = px.pie(
                names=list(sev_data.keys()),
                values=list(sev_data.values()),
                color=list(sev_data.keys()),
                color_discrete_map={"Urgent": "#C0392B", "High": "#E67E22", "Medium": "#F39C12"},
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No severity data.")

    modules = data.get("modules", {})
    if modules:
        st.subheader("Module Status")
        mod_data = []
        for mod_name, mod_info in modules.items():
            if isinstance(mod_info, dict):
                findings_n = mod_info.get("total_findings", mod_info.get("forward_issues", 0))
                mod_data.append({"Module": mod_name, "Findings": findings_n})
        if mod_data:
            df_mod = pd.DataFrame(mod_data)
            fig = px.bar(df_mod, x="Module", y="Findings", color="Findings",
                         color_continuous_scale="YlOrRd")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)


def _render_action_plan(action_plan: dict):
    st.subheader("Action Plan Tracker")

    for priority_key, label, color in [
        ("p1_urgent", "P1 - Urgent (24-48h)", "#C0392B"),
        ("p2_important", "P2 - Important (This Week)", "#E67E22"),
        ("p3_notice", "P3 - Notice (Two Weeks)", "#F39C12"),
    ]:
        actions = action_plan.get(priority_key, [])
        if not actions:
            continue

        st.markdown(f"### <span style='color:{color}'>{label} ({len(actions)} items)</span>",
                    unsafe_allow_html=True)

        df = pd.DataFrame(actions)
        display_cols = [c for c in ["ref_id", "description", "affected_subjects", "deadline"]
                        if c in df.columns]
        if display_cols:
            st.dataframe(df[display_cols], use_container_width=True, height=250)
        else:
            st.dataframe(df, use_container_width=True, height=250)


def _render_findings(findings: list[dict]):
    st.subheader("All Findings Explorer")

    if not findings:
        st.info("No findings data in JSON export.")
        return

    df = pd.DataFrame(findings)

    col1, col2, col3 = st.columns(3)
    with col1:
        severity_filter = st.multiselect(
            "Filter by Severity",
            options=sorted(df["severity"].unique().tolist()) if "severity" in df.columns else [],
            default=sorted(df["severity"].unique().tolist()) if "severity" in df.columns else [],
        )
    with col2:
        category_filter = st.multiselect(
            "Filter by Category",
            options=sorted(df["category"].unique().tolist()) if "category" in df.columns else [],
        )
    with col3:
        subject_filter = st.text_input("Filter by Subject ID (partial match)")

    filtered = df.copy()
    if severity_filter and "severity" in filtered.columns:
        filtered = filtered[filtered["severity"].isin(severity_filter)]
    if category_filter and "category" in filtered.columns:
        filtered = filtered[filtered["category"].isin(category_filter)]
    if subject_filter and "subject_id" in filtered.columns:
        filtered = filtered[filtered["subject_id"].str.contains(subject_filter, case=False, na=False)]

    st.caption(f"Showing {len(filtered)} of {len(df)} findings")
    st.dataframe(filtered, use_container_width=True, height=400)

    if "category" in filtered.columns and not filtered.empty:
        st.subheader("Category Distribution (Filtered)")
        cat_counts = filtered["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.treemap(cat_counts, path=["Category"], values="Count",
                         color="Count", color_continuous_scale="Reds")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def _render_oncology(data: dict):
    """Render oncology-specific results tab."""
    st.subheader("Oncology Analysis")

    modules = data.get("modules", {})

    recist = modules.get("recist", {})
    if recist:
        st.markdown("### RECIST 1.1 Summary")
        col1, col2, col3 = st.columns(3)
        rr_info = recist.get("response_rate", {})
        col1.metric("ORR", f"{rr_info.get('response_rate', 0):.1%}" if rr_info else "N/A")
        col2.metric("Evaluable", rr_info.get("evaluable_n", 0))
        col3.metric("Responders", rr_info.get("responders", 0))

        by_resp = rr_info.get("by_response", {})
        if by_resp:
            resp_df = pd.DataFrame([
                {"Response": k, "Count": v} for k, v in by_resp.items()
            ])
            fig = px.bar(resp_df, x="Response", y="Count",
                         color="Response",
                         color_discrete_map={"CR": "#2ca02c", "PR": "#1f77b4",
                                             "SD": "#ff7f0e", "PD": "#d62728", "NE": "#7f7f7f"})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    dose_esc = modules.get("dose_escalation", {})
    if dose_esc:
        st.markdown("### Dose Escalation")
        smc = dose_esc.get("smc_summary", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Enrolled", smc.get("total_enrolled", 0))
        col2.metric("Total DLTs", smc.get("total_dlt", 0))
        col3.metric("MTD Estimate", smc.get("mtd_estimate", "N/A"))

        cohort_summary = smc.get("cohort_summary", [])
        if cohort_summary:
            df_cohorts = pd.DataFrame(cohort_summary)
            st.dataframe(df_cohorts, use_container_width=True)

    hepatox = modules.get("hepatotoxicity", {})
    if hepatox:
        st.markdown("### Hepatotoxicity")
        hsummary = hepatox.get("summary", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Hy's Law Cases", hsummary.get("n_hys_law_met", 0))
        col2.metric("ALT >= 3xULN", hsummary.get("n_alt_elevated_3x", 0))
        col3.metric("Grade 3+ Events", hsummary.get("n_grade3_events", 0))


def _render_modules(modules: dict):
    st.subheader("Module Details")

    if not modules:
        st.info("No module data available.")
        return

    for mod_name, mod_info in modules.items():
        with st.expander(f"Module: {mod_name}", expanded=False):
            if isinstance(mod_info, dict):
                for k, v in mod_info.items():
                    if isinstance(v, (list, dict)):
                        st.json(v)
                    else:
                        st.write(f"**{k}**: {v}")
            else:
                st.write(mod_info)


def _render_ai(ai_results: dict):
    st.subheader("AI/LLM Enhancement Results")

    if not ai_results or not ai_results.get("enabled", False):
        st.info("AI enhancement was not enabled for this run.")
        st.markdown("""
        Enable AI by running the pipeline with `--ai` flag:
        ```bash
        python run_pipeline.py <edc.xlsx> --ai
        ```
        """)
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### AE Data Cleaning")
        cleaning = ai_results.get("ae_cleaning", {})
        if cleaning:
            st.metric("Processed", cleaning.get("total_processed", 0))
            st.metric("Corrections", cleaning.get("corrections", 0))
        else:
            st.write("No cleaning data.")

    with col2:
        st.markdown("#### AE Normalization")
        norm = ai_results.get("ae_normalization", {})
        if norm:
            st.metric("Unique Terms", norm.get("unique_terms", 0))
            st.metric("Normalized", norm.get("normalized", 0))
            st.metric("Variant Groups", norm.get("variant_groups", 0))
        else:
            st.write("No normalization data.")

    ab_results = ai_results.get("ab_test", {})
    if ab_results:
        st.markdown("---")
        st.markdown("#### A/B Test Results")
        st.json(ab_results)


def _render_export_section(data: dict):
    """Render export download buttons."""
    st.subheader("Export Results")

    col1, col2, col3, col4 = st.columns(4)

    json_str = json.dumps(data, ensure_ascii=False, default=str, indent=2)
    col1.download_button(
        "Download JSON",
        data=json_str,
        file_name="mm_results.json",
        mime="application/json",
    )

    col2.info("PDF/Word exports available via CLI")
    col3.info("Excel export available via CLI")
    col4.info("Use `python run_pipeline.py` for full exports")


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
