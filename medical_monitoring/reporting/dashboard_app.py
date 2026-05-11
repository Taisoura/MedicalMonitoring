"""
Streamlit Web Dashboard for Medical Monitoring.

Interactive real-time dashboard with:
- KPI summary cards
- Findings drill-down by severity/category/subject
- Action Plan tracker with status
- Cross-domain audit heatmaps
- CTCAE grade distribution
- Causality analysis overview
- AE quality radar

Launch:
    streamlit run medical_monitoring/reporting/dashboard_app.py -- --data mm_results.json

Or programmatically:
    from medical_monitoring.reporting.dashboard_app import launch_dashboard
    launch_dashboard(results_path="mm_results.json", port=8501)
"""

from __future__ import annotations

import json
import sys
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


def launch_dashboard(results_path: str | Path = "mm_results.json", port: int = 8501):
    """Launch the Streamlit dashboard programmatically."""
    import subprocess
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(Path(__file__).resolve()),
        "--server.port", str(port),
        "--", "--data", str(results_path),
    ]
    subprocess.Popen(cmd)


def load_results(path: str | Path) -> dict:
    """Load JSON results from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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

    # ── Sidebar ──────────────────────────────────────────────────────
    st.sidebar.title("Medical Monitoring")
    st.sidebar.markdown("---")

    data_path = st.sidebar.text_input(
        "Results JSON path",
        value="mm_results.json",
    )

    if not Path(data_path).exists():
        st.warning(f"Results file not found: {data_path}")
        st.info("Run the pipeline first, then export results as JSON:")
        st.code("python run_pipeline.py <edc.xlsx> --output-json mm_results.json")
        st.stop()

    data = load_results(data_path)

    # ── Header ───────────────────────────────────────────────────────
    st.title("Medical Monitoring Dashboard")
    meta = data.get("metadata", {})
    col1, col2, col3 = st.columns(3)
    col1.caption(f"Project: {meta.get('project', 'N/A')}")
    col2.caption(f"Generated: {data.get('generated_at', '')[:10]}")
    col3.caption(f"Schema: v{data.get('$version', '?')}")

    st.markdown("---")

    # ── KPI Cards ────────────────────────────────────────────────────
    summary = data.get("executive_summary", {})
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Findings", summary.get("total_findings", 0))
    kpi_cols[1].metric("Urgent (P1)", summary.get("urgent_count", 0),
                       delta=None if summary.get("urgent_count", 0) == 0 else "Action Required",
                       delta_color="inverse")
    kpi_cols[2].metric("High (P2)", summary.get("high_count", 0))
    kpi_cols[3].metric("Medium (P3)", summary.get("medium_count", 0))

    action_plan = data.get("action_plan", {})
    total_actions = (len(action_plan.get("p1_urgent", []))
                     + len(action_plan.get("p2_important", []))
                     + len(action_plan.get("p3_notice", [])))
    kpi_cols[4].metric("Total Actions", total_actions)

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
    tab_overview, tab_actions, tab_findings, tab_modules, tab_ai = st.tabs([
        "Overview", "Action Plan", "Findings", "Modules", "AI Results"
    ])

    # ── Tab: Overview ────────────────────────────────────────────────
    with tab_overview:
        _render_overview(data, summary)

    # ── Tab: Action Plan ─────────────────────────────────────────────
    with tab_actions:
        _render_action_plan(action_plan)

    # ── Tab: Findings ────────────────────────────────────────────────
    with tab_findings:
        _render_findings(data.get("findings", []))

    # ── Tab: Modules ─────────────────────────────────────────────────
    with tab_modules:
        _render_modules(data.get("modules", {}))

    # ── Tab: AI Results ──────────────────────────────────────────────
    with tab_ai:
        _render_ai(data.get("ai_results", {}))


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

    # Modules summary
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
            options=df["severity"].unique().tolist() if "severity" in df.columns else [],
            default=df["severity"].unique().tolist() if "severity" in df.columns else [],
        )
    with col2:
        category_filter = st.multiselect(
            "Filter by Category",
            options=df["category"].unique().tolist() if "category" in df.columns else [],
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
            types = cleaning.get("types", [])
            if types:
                st.write("Correction types:", ", ".join(str(t) for t in types))
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


# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
