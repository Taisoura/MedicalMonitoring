"""
Module B visualizations: Cross-domain audit results.

Charts:
    plot_findings_bar       -- Stacked bar of findings by rule & severity
    plot_findings_treemap   -- Treemap of finding counts by rule x severity
    plot_subject_rule_heatmap -- Subject x rule cross-heatmap
    plot_temporal_gantt     -- Temporal relationship Gantt (CM-AE dates)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..utils.helpers import Finding, findings_to_dataframe
from .base import VizBase


class AuditVisualizer(VizBase):
    """Visualization suite for cross-domain audit findings (Module B)."""

    # ── Stacked bar by rule & severity ────────────────────────────

    def plot_findings_bar(
        self,
        findings_by_rule: dict[str, list[Finding]],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Stacked bar chart of audit findings by rule, coloured by severity."""
        summary = self._summarize_by_rule_severity(findings_by_rule)

        if interactive:
            fig = self.create_plotly_figure()
            for sev in ["urgent", "high", "medium", "low"]:
                fig.add_trace(go.Bar(
                    x=summary["rule"],
                    y=summary.get(sev, [0] * len(summary["rule"])),
                    name=sev.capitalize(),
                    marker_color=self.color_for_severity(sev),
                ))
            fig.update_layout(
                barmode="stack",
                title=kwargs.get("title", "Audit Findings by Rule"),
                xaxis_title="Audit Rule", yaxis_title="Finding Count",
            )
            if output_dir:
                self.save_plotly(fig, "findings_stacked_bar", output_dir)
            return fig

        rules = summary["rule"]
        fig, ax = self.create_figure(figsize=(max(8, len(rules) * 1.2), 6))
        x = np.arange(len(rules))
        bottom = np.zeros(len(rules))

        for sev in ["low", "medium", "high", "urgent"]:
            vals = np.array(summary.get(sev, [0] * len(rules)), dtype=float)
            ax.bar(x, vals, bottom=bottom, label=sev.capitalize(),
                   color=self.color_for_severity(sev), width=0.6)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(rules, rotation=30, ha="right", fontsize=self.theme.tick_size)
        ax.set_ylabel("Finding Count")
        ax.set_title(kwargs.get("title", "Audit Findings by Rule"))
        ax.legend()

        if output_dir:
            self.save(fig, "findings_stacked_bar", output_dir)
        return fig

    # ── Treemap ───────────────────────────────────────────────────

    def plot_findings_treemap(
        self,
        findings_by_rule: dict[str, list[Finding]],
        *,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Treemap of findings -- area = count, colour = severity."""
        rows = []
        for rule_id, findings in findings_by_rule.items():
            for f in findings:
                rows.append({"rule": rule_id, "severity": f.severity, "count": 1})

        if not rows:
            fig = self.create_plotly_figure()
            fig.add_annotation(text="No findings", showarrow=False)
            return fig

        df = pd.DataFrame(rows).groupby(["rule", "severity"], as_index=False).sum()

        sev_map = self.severity_colors
        df["color"] = df["severity"].map(sev_map).fillna("#999999")

        fig = px.treemap(
            df, path=["rule", "severity"], values="count",
            color="severity",
            color_discrete_map=sev_map,
            title=kwargs.get("title", "Audit Findings Treemap"),
        )
        fig.update_layout(template=self.theme.plotly_template)
        if output_dir:
            self.save_plotly(fig, "findings_treemap", output_dir)
        return fig

    # ── Subject × rule heatmap ────────────────────────────────────

    def plot_subject_rule_heatmap(
        self,
        findings_by_rule: dict[str, list[Finding]],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Heatmap showing which subjects triggered which rules."""
        rows = []
        for rule_id, findings in findings_by_rule.items():
            for f in findings:
                if f.subject_id and f.subject_id != "MULTIPLE":
                    rows.append({"subject": f.subject_id, "rule": rule_id})

        if not rows:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No per-subject findings", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        df = pd.DataFrame(rows)
        pivot = df.groupby(["subject", "rule"]).size().unstack(fill_value=0)

        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Heatmap(
                z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
                colorscale="Reds",
            ))
            fig.update_layout(title=kwargs.get("title", "Subject × Rule Heatmap"))
            if output_dir:
                self.save_plotly(fig, "subject_rule_heatmap", output_dir)
            return fig

        import seaborn as sns
        h = max(4, len(pivot) * 0.3)
        fig, ax = self.create_figure(figsize=(max(8, len(pivot.columns) * 1.0), h))
        sns.heatmap(pivot, ax=ax, cmap="Reds", annot=len(pivot) <= 40, fmt="g",
                    linewidths=0.5)
        ax.set_title(kwargs.get("title", "Subject × Rule Heatmap"))

        if output_dir:
            self.save(fig, "subject_rule_heatmap", output_dir)
        return fig

    # ── Temporal Gantt ────────────────────────────────────────────

    def plot_temporal_gantt(
        self,
        temporal_data: pd.DataFrame,
        *,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """
        Gantt-style chart for CM-AE temporal relationships.

        temporal_data columns: subject_id, ae_name, ae_start, ae_end, cm_name, cm_start
        """
        if temporal_data.empty:
            fig = self.create_plotly_figure()
            fig.add_annotation(text="No temporal data", showarrow=False)
            return fig

        fig = self.create_plotly_figure()
        for i, row in temporal_data.iterrows():
            y_label = f"{row.get('subject_id', '')}: {row.get('ae_name', '')}"
            fig.add_trace(go.Bar(
                x=[row.get("ae_end", row.get("ae_start")) - row.get("ae_start")
                   if pd.notna(row.get("ae_end")) else pd.Timedelta(days=7)],
                y=[y_label],
                base=[row.get("ae_start")],
                orientation="h",
                marker_color=self.colors[0],
                name="AE" if i == 0 else None,
                showlegend=i == 0,
            ))

        fig.update_layout(
            title=kwargs.get("title", "CM-AE Temporal Relationships"),
            xaxis_title="Date", yaxis_title="Subject: AE",
            barmode="overlay",
        )
        if output_dir:
            self.save_plotly(fig, "temporal_gantt", output_dir)
        return fig

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _summarize_by_rule_severity(
        findings_by_rule: dict[str, list[Finding]],
    ) -> dict[str, list]:
        rules = sorted(findings_by_rule.keys())
        summary: dict[str, list] = {"rule": rules}
        for sev in ["urgent", "high", "medium", "low"]:
            summary[sev] = [
                sum(1 for f in findings_by_rule.get(r, []) if f.severity == sev)
                for r in rules
            ]
        return summary
