"""
Oncology-specific Visualizations

Generates publication-quality oncology clinical trial charts:
- Waterfall plot: best % change from baseline (tumor shrinkage)
- Spider plot: % change over time per subject
- Swimmer plot: treatment duration with response milestones
- DLT dot plot: DLT events by dose cohort
- Dose-toxicity heatmap: AE grade distribution across dose levels
- eDISH plot: ALT vs TBIL (xULN) quadrant plot

References:
  FDA oncology review templates, ASCO/ESMO poster standards
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


RESPONSE_COLORS = {
    "CR": "#2ca02c",    # green
    "PR": "#1f77b4",    # blue
    "SD": "#ff7f0e",    # orange
    "PD": "#d62728",    # red
    "NE": "#7f7f7f",    # gray
}

DOSE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def waterfall_plot(
    waterfall_data: list,
    output_path: str | Path | None = None,
    title: str = "Best Percentage Change from Baseline in Sum of Diameters",
    interactive: bool = False,
    theme: str = "academic_conference",
) -> Any:
    """Generate a waterfall plot of best tumor shrinkage."""
    if not waterfall_data:
        return None

    subjects = [w.subject_id for w in waterfall_data]
    changes = [w.best_pct_change for w in waterfall_data]
    responses = [getattr(w, "best_response", "NE") for w in waterfall_data]
    colors = [RESPONSE_COLORS.get(r, "#7f7f7f") for r in responses]

    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(len(subjects))),
            y=changes,
            marker_color=colors,
            text=[f"{s}<br>{r}<br>{c:.1f}%" for s, r, c in zip(subjects, responses, changes)],
            hovertemplate="%{text}<extra></extra>",
        ))
        fig.add_hline(y=-30, line_dash="dash", line_color="blue", annotation_text="PR threshold (-30%)")
        fig.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="PD threshold (+20%)")
        fig.update_layout(
            title=title,
            xaxis_title="Subject",
            yaxis_title="Best % Change from Baseline",
            showlegend=False,
            template="plotly_white",
        )
        if output_path:
            fig.write_html(str(output_path).replace(".png", ".html"))
        return fig

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 0.4), 6))
        bars = ax.bar(range(len(subjects)), changes, color=colors, edgecolor="white", linewidth=0.5)
        ax.axhline(y=-30, color="blue", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(y=20, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Subject", fontsize=10)
        ax.set_ylabel("Best % Change from Baseline", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels(subjects, rotation=90, fontsize=7)

        legend_elements = [
            mpatches.Patch(facecolor=RESPONSE_COLORS[r], label=r)
            for r in ["CR", "PR", "SD", "PD", "NE"] if r in set(responses)
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

        plt.tight_layout()
        if output_path:
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        return fig

    return None


def spider_plot(
    spider_data: list,
    output_path: str | Path | None = None,
    title: str = "Change from Baseline in Sum of Diameters Over Time",
    interactive: bool = False,
) -> Any:
    """Generate a spider plot showing tumor change trajectories."""
    if not spider_data:
        return None

    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        for entry in spider_data:
            fig.add_trace(go.Scatter(
                x=list(range(len(entry.pct_changes))),
                y=entry.pct_changes,
                mode="lines+markers",
                name=entry.subject_id,
                text=[f"{entry.subject_id}<br>{tp}<br>{pc:.1f}%"
                      for tp, pc in zip(entry.timepoints, entry.pct_changes)],
                hovertemplate="%{text}<extra></extra>",
                line=dict(width=1.5),
                marker=dict(size=4),
            ))
        fig.add_hline(y=-30, line_dash="dash", line_color="blue")
        fig.add_hline(y=20, line_dash="dash", line_color="red")
        fig.update_layout(
            title=title,
            xaxis_title="Assessment Timepoint",
            yaxis_title="% Change from Baseline",
            template="plotly_white",
        )
        if output_path:
            fig.write_html(str(output_path).replace(".png", ".html"))
        return fig

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(10, 6))
        for entry in spider_data:
            x = list(range(len(entry.pct_changes)))
            ax.plot(x, entry.pct_changes, "-o", markersize=3, linewidth=1, alpha=0.7,
                    label=entry.subject_id)
        ax.axhline(y=-30, color="blue", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(y=20, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Assessment Timepoint", fontsize=10)
        ax.set_ylabel("% Change from Baseline", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        if len(spider_data) <= 15:
            ax.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        if output_path:
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        return fig

    return None


def dlt_dot_plot(
    cohorts: dict,
    output_path: str | Path | None = None,
    title: str = "DLT Events by Dose Cohort",
    interactive: bool = False,
) -> Any:
    """Generate a DLT dot plot showing events per dose level."""
    if not cohorts:
        return None

    sorted_doses = sorted(cohorts.values(), key=lambda c: c.dose_mg or 0)
    dose_labels = [c.dose_level for c in sorted_doses]
    n_enrolled = [c.n_enrolled for c in sorted_doses]
    n_dlt = [c.n_dlt for c in sorted_doses]
    rates = [c.dlt_rate for c in sorted_doses]

    if interactive and HAS_PLOTLY:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=dose_labels, y=n_enrolled, name="Enrolled",
            marker_color="#1f77b4", opacity=0.4,
        ))
        fig.add_trace(go.Bar(
            x=dose_labels, y=n_dlt, name="DLT",
            marker_color="#d62728",
        ))
        fig.add_trace(go.Scatter(
            x=dose_labels, y=rates, name="DLT Rate",
            mode="lines+markers", marker=dict(size=10, color="#ff7f0e"),
            line=dict(width=2),
        ), secondary_y=True)
        fig.add_hline(y=0.30, line_dash="dot", line_color="orange",
                      secondary_y=True, annotation_text="Target 30%")
        fig.update_layout(
            title=title, barmode="overlay", template="plotly_white",
        )
        fig.update_yaxes(title_text="N Subjects", secondary_y=False)
        fig.update_yaxes(title_text="DLT Rate", tickformat=".0%", secondary_y=True)
        if output_path:
            fig.write_html(str(output_path).replace(".png", ".html"))
        return fig

    if HAS_MPL:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        x = np.arange(len(dose_labels))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, n_enrolled, width, label="Enrolled",
                        color="#1f77b4", alpha=0.4)
        bars2 = ax1.bar(x + width / 2, n_dlt, width, label="DLT",
                        color="#d62728")
        ax1.set_xlabel("Dose Level")
        ax1.set_ylabel("N Subjects")
        ax1.set_xticks(x)
        ax1.set_xticklabels(dose_labels, rotation=45, ha="right")

        ax2 = ax1.twinx()
        ax2.plot(x, rates, "o-", color="#ff7f0e", linewidth=2, markersize=8, label="DLT Rate")
        ax2.axhline(y=0.30, color="orange", linestyle=":", alpha=0.7)
        ax2.set_ylabel("DLT Rate")
        ax2.set_ylim(0, 1)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

        ax1.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        if output_path:
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        return fig

    return None


def edish_plot(
    edish_data: list,
    output_path: str | Path | None = None,
    title: str = "eDISH Plot: Peak ALT vs Peak TBIL (xULN)",
    interactive: bool = False,
) -> Any:
    """Generate eDISH (evaluation of Drug-Induced Serious Hepatotoxicity) plot."""
    if not edish_data:
        return None

    alt_vals = [e.alt_xuln for e in edish_data]
    tbil_vals = [e.tbil_xuln for e in edish_data]
    labels = [e.subject_id for e in edish_data]
    liver_mets = [e.has_liver_mets for e in edish_data]

    colors = ["#d62728" if lm else "#1f77b4" for lm in liver_mets]

    if interactive and HAS_PLOTLY:
        fig = go.Figure()
        for i, e in enumerate(edish_data):
            fig.add_trace(go.Scatter(
                x=[e.alt_xuln], y=[e.tbil_xuln],
                mode="markers",
                name=e.subject_id,
                marker=dict(
                    size=10,
                    color=colors[i],
                    symbol="diamond" if e.has_liver_mets else "circle",
                ),
                text=f"{e.subject_id}<br>ALT: {e.alt_xuln:.1f}xULN<br>TBIL: {e.tbil_xuln:.1f}xULN",
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ))
        fig.add_vline(x=3, line_dash="dash", line_color="gray")
        fig.add_hline(y=2, line_dash="dash", line_color="gray")
        fig.add_annotation(x=6, y=4, text="Hy's Law<br>Quadrant",
                           showarrow=False, font=dict(color="red", size=12))
        fig.update_layout(
            title=title,
            xaxis_title="Peak ALT (xULN)",
            yaxis_title="Peak TBIL (xULN)",
            xaxis=dict(type="log"),
            yaxis=dict(type="log"),
            template="plotly_white",
        )
        if output_path:
            fig.write_html(str(output_path).replace(".png", ".html"))
        return fig

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, e in enumerate(edish_data):
            marker = "D" if e.has_liver_mets else "o"
            ax.scatter(e.alt_xuln, e.tbil_xuln, c=colors[i], marker=marker, s=50,
                       alpha=0.7, edgecolors="black", linewidths=0.5)

        ax.axvline(x=3, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=2, color="gray", linestyle="--", alpha=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Peak ALT (xULN)", fontsize=10)
        ax.set_ylabel("Peak TBIL (xULN)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

        ax.text(6, 4, "Hy's Law\nQuadrant", color="red", fontsize=10,
                ha="center", style="italic")

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
                   markersize=8, label="No liver mets"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#d62728",
                   markersize=8, label="Liver mets"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")

        plt.tight_layout()
        if output_path:
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        return fig

    return None


def dose_toxicity_heatmap(
    ae_data: dict[str, Any],
    output_path: str | Path | None = None,
    title: str = "Adverse Event Grade Distribution by Dose Level",
    interactive: bool = False,
) -> Any:
    """
    Generate a dose-toxicity heatmap.

    ae_data format: {
        "dose_levels": ["50mg", "100mg", ...],
        "ae_terms": ["Nausea", "Fatigue", ...],
        "matrix": [[grade_counts...], ...],  # rows=AE, cols=dose
    }
    """
    if not ae_data or "matrix" not in ae_data:
        return None

    doses = ae_data.get("dose_levels", [])
    terms = ae_data.get("ae_terms", [])
    matrix = np.array(ae_data["matrix"])

    if interactive and HAS_PLOTLY:
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=doses,
            y=terms,
            colorscale="YlOrRd",
            text=matrix,
            texttemplate="%{text}",
            hovertemplate="AE: %{y}<br>Dose: %{x}<br>Count: %{z}<extra></extra>",
        ))
        fig.update_layout(
            title=title,
            xaxis_title="Dose Level",
            yaxis_title="Adverse Event",
            template="plotly_white",
        )
        if output_path:
            fig.write_html(str(output_path).replace(".png", ".html"))
        return fig

    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(max(6, len(doses) * 1.5), max(4, len(terms) * 0.4)))
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(np.arange(len(doses)))
        ax.set_yticks(np.arange(len(terms)))
        ax.set_xticklabels(doses, rotation=45, ha="right")
        ax.set_yticklabels(terms, fontsize=8)

        for i in range(len(terms)):
            for j in range(len(doses)):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, str(int(val)), ha="center", va="center",
                            color="white" if val > matrix.max() * 0.6 else "black",
                            fontsize=8)

        fig.colorbar(im, ax=ax, label="Count")
        ax.set_title(title, fontsize=12, fontweight="bold")
        plt.tight_layout()
        if output_path:
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        return fig

    return None
