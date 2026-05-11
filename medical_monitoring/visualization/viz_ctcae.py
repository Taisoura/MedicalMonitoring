"""
Module F2 visualizations: CTCAE grading engine results.

Charts:
    plot_edish          -- FDA eDISH plot (ALT/AST xULN vs TBIL xULN)
    plot_grade_waterfall -- CTCAE grade distribution waterfall by test
    plot_discrepancy_lollipop -- Grade discrepancy lollipop chart
    plot_lab_trend       -- Laboratory xULN/xLLN spaghetti with CTCAE bands
    plot_grade_waffle    -- Waffle chart of grade distribution
    plot_ncs_dot         -- NCS dispute dot plot by CTCAE grade
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..audit.ctcae_grading import CTCAEGradeResult, HysLawResult
from .base import VizBase


class CTCAEVisualizer(VizBase):
    """Visualization suite for CTCAE grading results (Module F2)."""

    # ── FDA eDISH Plot ────────────────────────────────────────────

    def plot_edish(
        self,
        hys_results: list[HysLawResult] | list[dict],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """
        FDA Composite eDISH Plot.

        x-axis: peak ALT or AST (xULN), y-axis: peak TBIL (xULN).
        Four quadrants: Normal | Temple's Corollary | Hy's Law | Cholestasis.
        """
        data = []
        for h in hys_results:
            d = h if isinstance(h, dict) else h.__dict__
            alt_fold = d.get("alt_fold") or 0
            ast_fold = d.get("ast_fold") or 0
            tbil_fold = d.get("tbil_fold") or 0
            x = max(alt_fold, ast_fold)
            if x > 0 or tbil_fold > 0:
                data.append({
                    "subject": d.get("subject_id", ""),
                    "x": x,
                    "y": tbil_fold,
                    "meets_hys": d.get("meets_hys_law", False),
                })

        if not data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No liver test data for eDISH", ha="center",
                    va="center", transform=ax.transAxes)
            return fig

        df = pd.DataFrame(data)

        if interactive:
            fig = self.create_plotly_figure()
            colours = [self.severity_colors["urgent"] if r else self.colors[1]
                       for r in df["meets_hys"]]
            fig.add_trace(go.Scatter(
                x=df["x"], y=df["y"], mode="markers+text",
                text=[s if m else "" for s, m in zip(df["subject"], df["meets_hys"])],
                textposition="top center",
                marker=dict(size=10, color=colours),
                hovertext=df["subject"],
            ))
            fig.add_vline(x=3, line_dash="dash", line_color="grey",
                          annotation_text="3xULN")
            fig.add_hline(y=2, line_dash="dash", line_color="grey",
                          annotation_text="2xULN")

            fig.add_annotation(x=1.5, y=0.8, text="Normal", showarrow=False,
                               font=dict(color="grey"))
            fig.add_annotation(x=8, y=0.8, text="Temple's Corollary", showarrow=False,
                               font=dict(color="grey"))
            fig.add_annotation(x=8, y=4, text="Hy's Law", showarrow=False,
                               font=dict(color="red", size=14))
            fig.add_annotation(x=1.5, y=4, text="Cholestasis /\nGilbert's",
                               showarrow=False, font=dict(color="grey"))

            fig.update_layout(
                title=kwargs.get("title", "eDISH Plot (Drug-Induced Liver Injury)"),
                xaxis_title="Peak ALT or AST (xULN)",
                yaxis_title="Peak TBIL (xULN)",
                xaxis_type="log", yaxis_type="log",
            )
            if output_dir:
                self.save_plotly(fig, "edish_plot", output_dir)
            return fig

        fig, ax = self.create_figure(figsize=(9, 8))
        for _, row in df.iterrows():
            c = self.severity_colors["urgent"] if row["meets_hys"] else self.colors[1]
            ax.scatter(row["x"], row["y"], c=c, s=80, zorder=3, edgecolors="k", linewidth=0.5)
            if row["meets_hys"]:
                ax.annotate(row["subject"], (row["x"], row["y"]),
                            fontsize=self.theme.annotation_size, ha="center", va="bottom")

        ax.axvline(3, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(2, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.text(1.2, 0.5, "Normal", color="grey", fontsize=self.theme.annotation_size)
        ax.text(6, 0.5, "Temple's Corollary", color="grey",
                fontsize=self.theme.annotation_size)
        ax.text(6, 4, "Hy's Law", color="red", fontsize=self.theme.label_size,
                fontweight="bold")
        ax.text(1.2, 4, "Cholestasis/\nGilbert's", color="grey",
                fontsize=self.theme.annotation_size)

        ax.set_xlabel("Peak ALT or AST (xULN)")
        ax.set_ylabel("Peak TBIL (xULN)")
        ax.set_title(kwargs.get("title", "eDISH Plot (Drug-Induced Liver Injury)"))

        if output_dir:
            self.save(fig, "edish_plot", output_dir)
        return fig

    # ── Grade distribution waterfall ──────────────────────────────

    def plot_grade_waterfall(
        self,
        graded_records: list[CTCAEGradeResult] | list[dict],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Waterfall of CTCAE grades grouped by test code."""
        records = []
        for r in graded_records:
            d = r if isinstance(r, dict) else r.__dict__
            records.append(d)

        if not records:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No graded records", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        df = pd.DataFrame(records)
        test_groups = df.groupby("test_code")["ctcae_grade"].value_counts().unstack(fill_value=0)
        tests = test_groups.index.tolist()
        grades = sorted([c for c in test_groups.columns if isinstance(c, int)])

        if interactive:
            fig = self.create_plotly_figure()
            for g in grades:
                fig.add_trace(go.Bar(
                    x=tests,
                    y=test_groups.get(g, pd.Series(0, index=tests)).values,
                    name=f"Grade {g}",
                    marker_color=self.color_for_grade(g),
                ))
            fig.update_layout(
                barmode="stack",
                title=kwargs.get("title", "CTCAE Grade Distribution by Test"),
                yaxis_title="Count",
            )
            if output_dir:
                self.save_plotly(fig, "grade_waterfall", output_dir)
            return fig

        fig, ax = self.create_figure(figsize=(max(8, len(tests) * 0.8), 6))
        x = np.arange(len(tests))
        bottom = np.zeros(len(tests))
        for g in grades:
            vals = test_groups.get(g, pd.Series(0, index=tests)).values.astype(float)
            ax.bar(x, vals, bottom=bottom, color=self.color_for_grade(g),
                   label=f"Grade {g}", width=0.6)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(tests, rotation=30, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(kwargs.get("title", "CTCAE Grade Distribution by Test"))
        ax.legend()

        if output_dir:
            self.save(fig, "grade_waterfall", output_dir)
        return fig

    # ── Discrepancy lollipop ──────────────────────────────────────

    def plot_discrepancy_lollipop(
        self,
        discrepancies: list[CTCAEGradeResult] | list[dict],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Lollipop chart of grade discrepancies (CTCAE vs AE recorded)."""
        data = []
        for r in discrepancies:
            d = r if isinstance(r, dict) else r.__dict__
            data.append(d)

        if not data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No discrepancies", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        df = pd.DataFrame(data)
        df["label"] = df["subject_id"] + " / " + df["test_code"]
        df = df.sort_values("grade_discrepancy", ascending=True).head(30)

        if interactive:
            fig = self.create_plotly_figure()
            colours = [self.severity_colors["urgent"] if d >= 2
                       else self.severity_colors["high"]
                       for d in df["grade_discrepancy"]]
            fig.add_trace(go.Scatter(
                x=df["grade_discrepancy"], y=df["label"], mode="markers",
                marker=dict(size=12, color=colours),
            ))
            for _, row in df.iterrows():
                fig.add_shape(
                    type="line", x0=0, x1=row["grade_discrepancy"],
                    y0=row["label"], y1=row["label"],
                    line=dict(color="#CCCCCC", width=1),
                )
            fig.update_layout(
                title=kwargs.get("title", "CTCAE Grade Discrepancies"),
                xaxis_title="Grade Discrepancy (CTCAE - AE)",
            )
            if output_dir:
                self.save_plotly(fig, "discrepancy_lollipop", output_dir)
            return fig

        fig, ax = self.create_figure(figsize=(8, max(4, len(df) * 0.3)))
        y = np.arange(len(df))
        colours = [self.severity_colors["urgent"] if d >= 2
                   else self.severity_colors["high"]
                   for d in df["grade_discrepancy"]]
        ax.hlines(y, 0, df["grade_discrepancy"].values, color="#CCCCCC", linewidth=1)
        ax.scatter(df["grade_discrepancy"].values, y, c=colours, s=80, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(df["label"].values, fontsize=max(5, self.theme.tick_size - 2))
        ax.set_xlabel("Grade Discrepancy (CTCAE - AE)")
        ax.set_title(kwargs.get("title", "CTCAE Grade Discrepancies"))
        ax.axvline(0, color="black", linewidth=0.5)

        if output_dir:
            self.save(fig, "discrepancy_lollipop", output_dir)
        return fig

    # ── Lab xULN/xLLN trend ──────────────────────────────────────

    def plot_lab_trend(
        self,
        graded_records: list[CTCAEGradeResult] | list[dict],
        test_code: str,
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Spaghetti of lab values (xULN) with CTCAE grade threshold bands."""
        data = []
        for r in graded_records:
            d = r if isinstance(r, dict) else r.__dict__
            if d.get("test_code", "").upper() == test_code.upper():
                data.append(d)

        if not data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, f"No data for {test_code}", ha="center",
                    va="center", transform=ax.transAxes)
            return fig

        df = pd.DataFrame(data)
        if "upper_limit" in df.columns:
            df["fold"] = df["result"] / df["upper_limit"].replace(0, np.nan)
        else:
            df["fold"] = df["result"]

        subjects = df["subject_id"].unique()

        if interactive:
            fig = self.create_plotly_figure()
            for subj in subjects:
                sdf = df[df["subject_id"] == subj].sort_values("visit")
                fig.add_trace(go.Scatter(
                    x=sdf["visit"], y=sdf["fold"], mode="lines+markers",
                    name=subj, line=dict(width=1),
                ))
            for g, thr in [(1, 1), (2, 3), (3, 5), (4, 20)]:
                fig.add_hline(y=thr, line_dash="dot",
                              line_color=self.color_for_grade(g),
                              annotation_text=f"G{g}")
            fig.update_layout(
                title=kwargs.get("title", f"{test_code} Lab Values (xULN)"),
                yaxis_title="Fold of ULN",
            )
            if output_dir:
                self.save_plotly(fig, f"lab_trend_{test_code}", output_dir)
            return fig

        fig, ax = self.create_figure()
        for subj in subjects:
            sdf = df[df["subject_id"] == subj].sort_values("visit")
            ax.plot(sdf["visit"], sdf["fold"], "-o", alpha=0.5, markersize=3)

        for g, thr in [(1, 1), (2, 3), (3, 5), (4, 20)]:
            ax.axhline(thr, color=self.color_for_grade(g), linestyle=":",
                       linewidth=0.8, label=f"G{g} ({thr}xULN)")

        ax.set_xlabel("Visit")
        ax.set_ylabel("Fold of ULN")
        ax.set_title(kwargs.get("title", f"{test_code} Lab Values (xULN)"))
        ax.legend(fontsize=self.theme.legend_fontsize, loc="upper right")

        if output_dir:
            self.save(fig, f"lab_trend_{test_code}", output_dir)
        return fig

    # ── Grade waffle chart ────────────────────────────────────────

    def plot_grade_waffle(
        self,
        grade_distribution: dict[int, int],
        *,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """Waffle chart (10x10 grid) showing grade proportions."""
        total = sum(grade_distribution.values())
        if total == 0:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        proportions = {g: max(1, round(100 * c / total)) for g, c in grade_distribution.items() if c > 0}
        adj_total = sum(proportions.values())
        if adj_total > 100:
            largest_key = max(proportions, key=proportions.get)
            proportions[largest_key] -= (adj_total - 100)

        grid = np.zeros(100, dtype=int)
        idx = 0
        for grade in sorted(proportions.keys()):
            count = proportions[grade]
            grid[idx:idx + count] = grade
            idx += count

        grid = grid.reshape(10, 10)

        fig, ax = self.create_figure(figsize=(7, 7))
        for i in range(10):
            for j in range(10):
                g = grid[i, j]
                rect = plt.Rectangle((j, 9 - i), 0.9, 0.9,
                                     facecolor=self.color_for_grade(g),
                                     edgecolor="white", linewidth=1)
                ax.add_patch(rect)

        ax.set_xlim(-0.1, 10.1)
        ax.set_ylim(-0.1, 10.1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(kwargs.get("title", "CTCAE Grade Distribution (Waffle)"),
                     fontsize=self.theme.title_size)

        legend_patches = [
            mpatches.Patch(color=self.color_for_grade(g), label=f"Grade {g} ({c})")
            for g, c in sorted(grade_distribution.items()) if c > 0
        ]
        ax.legend(handles=legend_patches, loc="lower right",
                  fontsize=self.theme.legend_fontsize)

        if output_dir:
            self.save(fig, "grade_waffle", output_dir)
        return fig

    # ── NCS dispute dot plot ──────────────────────────────────────

    def plot_ncs_dot(
        self,
        ncs_disputes: list[CTCAEGradeResult] | list[dict],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Dot plot of NCS (no clinical significance) disputes by CTCAE grade."""
        data = []
        for r in ncs_disputes:
            d = r if isinstance(r, dict) else r.__dict__
            data.append(d)

        if not data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No NCS disputes", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        df = pd.DataFrame(data)
        grade_counts = df.groupby("ctcae_grade").size().reset_index(name="count")

        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Scatter(
                x=grade_counts["ctcae_grade"],
                y=grade_counts["count"],
                mode="markers",
                marker=dict(
                    size=grade_counts["count"] * 5 + 10,
                    color=[self.color_for_grade(g) for g in grade_counts["ctcae_grade"]],
                ),
                text=[f"G{g}: {c} disputes" for g, c in
                      zip(grade_counts["ctcae_grade"], grade_counts["count"])],
            ))
            fig.update_layout(
                title=kwargs.get("title", "NCS Disputes by CTCAE Grade"),
                xaxis_title="CTCAE Grade", yaxis_title="Number of Disputes",
            )
            if output_dir:
                self.save_plotly(fig, "ncs_dot_plot", output_dir)
            return fig

        fig, ax = self.create_figure()
        for _, row in grade_counts.iterrows():
            g = row["ctcae_grade"]
            c = row["count"]
            ax.scatter(g, c, s=c * 50 + 80, color=self.color_for_grade(g),
                       zorder=3, edgecolors="k")
            ax.annotate(f"{c}", (g, c), ha="center", va="center",
                        fontsize=self.theme.label_size, fontweight="bold")

        ax.set_xticks(sorted(grade_counts["ctcae_grade"].unique()))
        ax.set_xlabel("CTCAE Grade")
        ax.set_ylabel("Number of NCS Disputes")
        ax.set_title(kwargs.get("title", "NCS Disputes by CTCAE Grade"))

        if output_dir:
            self.save(fig, "ncs_dot_plot", output_dir)
        return fig
