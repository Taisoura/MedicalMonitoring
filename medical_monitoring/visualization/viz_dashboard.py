"""
Executive Dashboard: one-page summary of all Medical Monitoring results.

Produces:
    - Static: matplotlib multi-panel A3 PDF / PNG
    - Interactive: Plotly HTML with drill-down
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..utils.helpers import Finding
from .base import VizBase


class DashboardVisualizer(VizBase):
    """One-page executive dashboard combining KPIs from all modules."""

    def generate_dashboard(
        self,
        pipeline_results: dict[str, Any],
        subject_db: pd.DataFrame | None = None,
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """
        Generate the complete executive dashboard.

        pipeline_results: the dict returned by MedicalMonitoringPipeline.run()
        """
        if interactive:
            return self._dashboard_plotly(pipeline_results, subject_db,
                                          output_dir=output_dir, **kwargs)
        return self._dashboard_matplotlib(pipeline_results, subject_db,
                                          output_dir=output_dir, **kwargs)

    # ── matplotlib static dashboard ───────────────────────────────

    def _dashboard_matplotlib(
        self,
        results: dict[str, Any],
        subject_db: pd.DataFrame | None,
        *,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> plt.Figure:
        self._apply_theme()
        fig = plt.figure(figsize=(20, 14))
        fig.patch.set_facecolor(self.theme.background_color)
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

        # ── Row 0: KPI cards ──
        ax_kpi = fig.add_subplot(gs[0, :2])
        self._draw_kpi_cards(ax_kpi, results)

        ax_sev = fig.add_subplot(gs[0, 2:])
        self._draw_severity_bar(ax_sev, results)

        # ── Row 1: AE quality radar + top actions ──
        ax_radar = fig.add_subplot(gs[1, :2], projection="polar")
        self._draw_ae_radar(ax_radar, results)

        ax_actions = fig.add_subplot(gs[1, 2:])
        self._draw_top_actions(ax_actions, results)

        # ── Row 2: eDISH mini + Forest mini ──
        ax_edish = fig.add_subplot(gs[2, :2])
        self._draw_edish_mini(ax_edish, results)

        ax_forest = fig.add_subplot(gs[2, 2:])
        self._draw_forest_mini(ax_forest, results)

        fig.suptitle(
            kwargs.get("title", "Medical Monitoring Executive Dashboard"),
            fontsize=self.theme.title_size + 4,
            fontweight="bold", y=0.98,
        )

        if output_dir:
            self.save(fig, "executive_dashboard", output_dir, formats=("png", "pdf"))
        return fig

    def _draw_kpi_cards(self, ax: plt.Axes, results: dict) -> None:
        ax.axis("off")
        report = results.get("report", {})
        summary = report.get("executive_summary", {})
        stats = report.get("statistics", {})

        kpis = [
            ("Total Findings", str(summary.get("total_findings", 0)), self.colors[0]),
            ("Urgent (P1)", str(summary.get("urgent_count", 0)), self.severity_colors["urgent"]),
            ("High (P2)", str(summary.get("high_count", 0)), self.severity_colors["high"]),
            ("Actions", str(stats.get("total_actions", 0)), self.colors[2]),
        ]

        for i, (label, value, color) in enumerate(kpis):
            x = 0.12 + i * 0.22
            ax.text(x, 0.7, value, ha="center", va="center",
                    fontsize=28, fontweight="bold", color=color,
                    transform=ax.transAxes)
            ax.text(x, 0.3, label, ha="center", va="center",
                    fontsize=self.theme.label_size, color="#666666",
                    transform=ax.transAxes)

        ax.set_title("Key Performance Indicators", fontsize=self.theme.title_size)

    def _draw_severity_bar(self, ax: plt.Axes, results: dict) -> None:
        all_findings = self._collect_findings(results)
        sev_counts = {"urgent": 0, "high": 0, "medium": 0, "low": 0}
        for f in all_findings:
            sev = f.severity if hasattr(f, "severity") else "medium"
            sev_counts[sev] = sev_counts.get(sev, 0) + 1

        labels = list(sev_counts.keys())
        values = list(sev_counts.values())
        colours = [self.color_for_severity(s) for s in labels]
        ax.barh(labels, values, color=colours, height=0.6)
        ax.set_xlabel("Count")
        ax.set_title("Findings by Severity")
        for spine in ax.spines.values():
            spine.set_visible(self.theme.spine_visible)

    def _draw_ae_radar(self, ax: plt.Axes, results: dict) -> None:
        ae = results.get("ae_quality", {})
        if not ae or "error" in ae:
            ax.set_title("AE Quality (N/A)")
            return

        dims = ["Naming", "Dates", "Outcomes", "Severity", "SAE",
                "Drug", "Death", "NCS", "XRef", "Actions"]
        keys = [f"{i}_{d.lower()}" for i, d in enumerate(["naming", "dates", "outcomes",
                "severity", "sae_gaps", "drug_action", "death", "ncs",
                "cross_ref", "action_plan"], 1)]

        total = max(ae.get("summary", {}).get("total_ae_records", 1), 1)
        scores = []
        for k in keys:
            v = ae.get(k, [])
            n = len(v) if isinstance(v, list) else 0
            scores.append(max(0, 1 - n / total))

        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
        vals = scores + [scores[0]]
        angles_closed = angles + [angles[0]]
        ax.fill(angles_closed, vals, color=self.colors[1], alpha=0.3)
        ax.plot(angles_closed, vals, color=self.colors[0], linewidth=2)
        ax.set_xticks(angles)
        ax.set_xticklabels(dims, fontsize=7)
        ax.set_ylim(0, 1)
        ax.set_title("AE Quality Radar", fontsize=self.theme.label_size, pad=15)

    def _draw_top_actions(self, ax: plt.Axes, results: dict) -> None:
        ax.axis("off")
        report = results.get("report", {})
        actions = report.get("action_plan", {}).get("recommendations", [])[:5]

        if not actions:
            ax.text(0.5, 0.5, "No action items", ha="center", va="center",
                    transform=ax.transAxes, fontsize=self.theme.label_size)
            ax.set_title("Top Action Items")
            return

        header = ["Priority", "Description", "Deadline"]
        rows = []
        for a in actions:
            if hasattr(a, "priority"):
                rows.append([a.priority.upper(), a.description[:50], a.deadline])
            elif isinstance(a, dict):
                rows.append([a.get("priority", "").upper(),
                             a.get("description", "")[:50],
                             a.get("deadline", "")])

        if rows:
            table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="left")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.0, 1.4)

        ax.set_title("Top Action Items", fontsize=self.theme.title_size)

    def _draw_edish_mini(self, ax: plt.Axes, results: dict) -> None:
        ctcae = results.get("ctcae_grading", {})
        hys = ctcae.get("hys_law_screening", [])

        if not hys:
            ax.text(0.5, 0.5, "No liver test data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("eDISH (mini)")
            return

        for h in hys:
            d = h if isinstance(h, dict) else h.__dict__
            alt_fold = d.get("alt_fold") or 0
            ast_fold = d.get("ast_fold") or 0
            tbil_fold = d.get("tbil_fold") or 0
            x = max(alt_fold, ast_fold)
            meets = d.get("meets_hys_law", False)
            c = self.severity_colors["urgent"] if meets else self.colors[1]
            if x > 0 or tbil_fold > 0:
                ax.scatter(x, tbil_fold, c=c, s=40, edgecolors="k", linewidth=0.3)

        ax.axvline(3, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.axhline(2, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.set_xlabel("ALT/AST (xULN)", fontsize=8)
        ax.set_ylabel("TBIL (xULN)", fontsize=8)
        ax.set_title("eDISH (mini)", fontsize=self.theme.label_size)

    def _draw_forest_mini(self, ax: plt.Axes, results: dict) -> None:
        stats = results.get("statistics", {})
        or_data = stats.get("mortality_or", [])

        if not or_data:
            ax.text(0.5, 0.5, "No OR data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Forest Plot (mini)")
            return

        valid = [d for d in or_data if np.isfinite(d.get("odds_ratio", float("nan")))][:8]
        if not valid:
            ax.text(0.5, 0.5, "No finite ORs", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("Forest Plot (mini)")
            return

        valid.sort(key=lambda x: x.get("odds_ratio", 1))
        y = np.arange(len(valid))
        for i, d in enumerate(valid):
            orv = d["odds_ratio"]
            lo = d.get("ci_lower", orv * 0.5)
            hi = d.get("ci_upper", orv * 2)
            pv = d.get("p_value", 1)
            c = self.colors[0] if pv < 0.05 else "#AAAAAA"
            ax.plot([lo, hi], [i, i], color=c, linewidth=1.5)
            ax.plot(orv, i, "D", color=c, markersize=5)

        ax.axvline(1, color="grey", linestyle="--", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([d.get("predictor", "")[:20] for d in valid], fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("OR (95% CI)", fontsize=8)
        ax.set_title("Forest Plot (mini)", fontsize=self.theme.label_size)

    # ── Plotly interactive dashboard ──────────────────────────────

    def _dashboard_plotly(
        self,
        results: dict[str, Any],
        subject_db: pd.DataFrame | None,
        *,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "Findings by Severity", "AE Quality Radar",
                "eDISH Plot", "Grade Distribution",
                "Top Findings Categories", "Action Priority",
            ],
            specs=[
                [{"type": "bar"}, {"type": "polar"}],
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}],
            ],
            vertical_spacing=0.1, horizontal_spacing=0.08,
        )

        # (1,1) Severity bar
        all_findings = self._collect_findings(results)
        sev_counts = {"urgent": 0, "high": 0, "medium": 0, "low": 0}
        for f in all_findings:
            sev = f.severity if hasattr(f, "severity") else "medium"
            sev_counts[sev] = sev_counts.get(sev, 0) + 1

        fig.add_trace(go.Bar(
            x=list(sev_counts.keys()), y=list(sev_counts.values()),
            marker_color=[self.color_for_severity(s) for s in sev_counts],
        ), row=1, col=1)

        # (1,2) AE quality radar
        ae = results.get("ae_quality", {})
        dims = ["Naming", "Dates", "Outcomes", "Severity", "SAE",
                "Drug", "Death", "NCS", "XRef", "Actions"]
        keys = ["1_naming", "2_dates", "3_outcomes", "4_severity",
                "5_sae_gaps", "6_drug_action", "7_death", "8_ncs",
                "9_cross_ref", "10_action_plan"]
        total = max(ae.get("summary", {}).get("total_ae_records", 1), 1) if ae else 1
        scores = [max(0, 1 - len(ae.get(k, [])) / total) if isinstance(ae.get(k, []), list) else 1
                  for k in keys]
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]], theta=dims + [dims[0]],
            fill="toself", line=dict(color=self.colors[0]),
        ), row=1, col=2)

        # (2,1) eDISH mini
        hys = results.get("ctcae_grading", {}).get("hys_law_screening", [])
        edish_x, edish_y, edish_c = [], [], []
        for h in hys:
            d = h if isinstance(h, dict) else h.__dict__
            x = max(d.get("alt_fold") or 0, d.get("ast_fold") or 0)
            y = d.get("tbil_fold") or 0
            if x > 0 or y > 0:
                edish_x.append(x)
                edish_y.append(y)
                edish_c.append(self.severity_colors["urgent"]
                               if d.get("meets_hys_law") else self.colors[1])
        fig.add_trace(go.Scatter(
            x=edish_x, y=edish_y, mode="markers",
            marker=dict(color=edish_c, size=8),
        ), row=2, col=1)

        # (2,2) CTCAE grade distribution
        ctcae_sum = results.get("ctcae_grading", {}).get("summary", {})
        g_dist = ctcae_sum.get("grade_distribution", {})
        if g_dist:
            fig.add_trace(go.Bar(
                x=[f"G{g}" for g in sorted(g_dist.keys())],
                y=[g_dist[g] for g in sorted(g_dist.keys())],
                marker_color=[self.color_for_grade(g) for g in sorted(g_dist.keys())],
            ), row=2, col=2)

        # (3,1) top finding categories
        cat_counts = {}
        for f in all_findings:
            cat = f.category if hasattr(f, "category") else "Unknown"
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
        top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:8]
        if top_cats:
            fig.add_trace(go.Bar(
                x=[c[0][:25] for c in top_cats],
                y=[c[1] for c in top_cats],
                marker_color=self.colors[0],
            ), row=3, col=1)

        # (3,2) action priority counts
        report = results.get("report", {})
        ap = report.get("action_plan", {}).get("recommendations", [])
        prio_counts = {"P1": 0, "P2": 0, "P3": 0}
        for a in ap:
            p = a.priority if hasattr(a, "priority") else a.get("priority", "P3")
            prio_counts[p] = prio_counts.get(p, 0) + 1
        fig.add_trace(go.Bar(
            x=list(prio_counts.keys()), y=list(prio_counts.values()),
            marker_color=[self.severity_colors["urgent"],
                          self.severity_colors["high"],
                          self.severity_colors["medium"]],
        ), row=3, col=2)

        fig.update_layout(
            title_text=kwargs.get("title", "Medical Monitoring Executive Dashboard"),
            showlegend=False,
            height=1000,
            template=self.theme.plotly_template,
        )

        if output_dir:
            self.save_plotly(fig, "executive_dashboard", output_dir, as_html=True)
        return fig

    # ── Helpers ────────────────────────────────────────────────────

    @staticmethod
    def _collect_findings(results: dict) -> list[Finding]:
        """Recursively collect all Finding objects from pipeline results."""
        findings: list[Finding] = []

        def _recurse(obj: Any) -> None:
            if isinstance(obj, Finding):
                findings.append(obj)
            elif isinstance(obj, list):
                for item in obj:
                    _recurse(item)
            elif isinstance(obj, dict):
                for v in obj.values():
                    _recurse(v)

        for key in ["cross_domain", "causality", "ae_quality"]:
            _recurse(results.get(key, {}))

        ctcae = results.get("ctcae_grading", {})
        if ctcae:
            from ..audit.ctcae_grading import CTCAEGradingEngine
            engine = CTCAEGradingEngine()
            findings.extend(engine.generate_findings(ctcae))

        return findings
