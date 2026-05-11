"""
Module E visualizations: Drug causality audit results.

Charts:
    plot_sankey            -- IB risk category → AE term Sankey flow
    plot_bidirectional_waterfall -- Forward (under) vs reverse (over) findings
    plot_confounding_radar -- Multi-dimensional confounding factor radar
    plot_consistency_heatmap -- AE term × causality assessment consistency
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..utils.helpers import Finding
from .base import VizBase


class CausalityVisualizer(VizBase):
    """Visualization suite for drug causality audit (Module E)."""

    # ── Sankey: IB risk → AE mapping ──────────────────────────────

    def plot_sankey(
        self,
        causality_results: dict[str, Any],
        *,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Sankey diagram showing IB risk categories flowing to AE findings."""
        fwd = causality_results.get("forward_findings", [])
        rev = causality_results.get("reverse_findings", [])
        all_findings: list[Finding] = fwd + rev

        if not all_findings:
            fig = self.create_plotly_figure()
            fig.add_annotation(text="No causality findings", showarrow=False)
            return fig

        sources_set: list[str] = []
        targets_set: list[str] = []
        link_values: list[int] = []

        cat_count: dict[tuple[str, str], int] = defaultdict(int)
        for f in all_findings:
            cat = f.category.split("(")[-1].rstrip(")") if "(" in f.category else f.category
            subj = f.subject_id
            cat_count[(cat, subj)] += 1

        label_list: list[str] = []
        source_indices: list[int] = []
        target_indices: list[int] = []
        values: list[int] = []

        categories = sorted(set(k[0] for k in cat_count))
        subjects = sorted(set(k[1] for k in cat_count))

        label_list = categories + subjects
        cat_idx = {c: i for i, c in enumerate(categories)}
        subj_idx = {s: i + len(categories) for i, s in enumerate(subjects)}

        for (cat, subj), cnt in cat_count.items():
            source_indices.append(cat_idx[cat])
            target_indices.append(subj_idx[subj])
            values.append(cnt)

        fig = self.create_plotly_figure()
        fig.add_trace(go.Sankey(
            node=dict(
                label=label_list,
                color=self.colors[:len(label_list)],
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
            ),
        ))
        fig.update_layout(title=kwargs.get("title", "IB Risk → Subject Causality Flow"))
        if output_dir:
            self.save_plotly(fig, "causality_sankey", output_dir)
        return fig

    # ── Bidirectional waterfall ────────────────────────────────────

    def plot_bidirectional_waterfall(
        self,
        causality_results: dict[str, Any],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """
        Waterfall chart: forward findings (under-reporting) go downward,
        reverse findings (over-reporting) go upward.
        """
        fwd = causality_results.get("forward_findings", [])
        rev = causality_results.get("reverse_findings", [])

        fwd_cats = Counter(f.category for f in fwd)
        rev_cats = Counter(f.category for f in rev)
        all_cats = sorted(set(list(fwd_cats.keys()) + list(rev_cats.keys())))

        fwd_vals = [fwd_cats.get(c, 0) for c in all_cats]
        rev_vals = [rev_cats.get(c, 0) for c in all_cats]

        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Bar(
                x=all_cats, y=[-v for v in fwd_vals],
                name="Forward (under-reported)",
                marker_color=self.severity_colors["high"],
            ))
            fig.add_trace(go.Bar(
                x=all_cats, y=rev_vals,
                name="Reverse (over-reported)",
                marker_color=self.colors[1],
            ))
            fig.update_layout(
                barmode="relative",
                title=kwargs.get("title", "Bidirectional Causality Audit"),
                yaxis_title="Finding Count",
            )
            if output_dir:
                self.save_plotly(fig, "causality_waterfall", output_dir)
            return fig

        fig, ax = self.create_figure()
        x = np.arange(len(all_cats))
        ax.bar(x, [-v for v in fwd_vals], color=self.severity_colors["high"],
               label="Forward (under-reported)", width=0.5)
        ax.bar(x, rev_vals, color=self.colors[1],
               label="Reverse (over-reported)", width=0.5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_cats, rotation=30, ha="right", fontsize=self.theme.tick_size)
        ax.set_ylabel("Finding Count")
        ax.set_title(kwargs.get("title", "Bidirectional Causality Audit"))
        ax.legend()

        if output_dir:
            self.save(fig, "causality_waterfall", output_dir)
        return fig

    # ── Confounding factor radar ──────────────────────────────────

    def plot_confounding_radar(
        self,
        scores_by_subject: dict[str, list[dict]],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """
        Radar chart of confounding factor scores per subject.

        scores_by_subject: {subject_id: [{"factor_name": ..., "score": ...}, ...]}
        """
        if not scores_by_subject:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No confounding data", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        all_factors = sorted(set(
            s["factor_name"]
            for scores in scores_by_subject.values()
            for s in scores
        ))

        if interactive:
            fig = self.create_plotly_figure()
            for i, (subj, scores) in enumerate(list(scores_by_subject.items())[:8]):
                score_map = {s["factor_name"]: s["score"] for s in scores}
                vals = [score_map.get(f, 0) for f in all_factors]
                fig.add_trace(go.Scatterpolar(
                    r=vals + [vals[0]],
                    theta=all_factors + [all_factors[0]],
                    name=subj,
                    line=dict(color=self.colors[i % len(self.colors)]),
                ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                title=kwargs.get("title", "Confounding Factor Scores"),
            )
            if output_dir:
                self.save_plotly(fig, "confounding_radar", output_dir)
            return fig

        angles = np.linspace(0, 2 * np.pi, len(all_factors), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = self.create_figure(figsize=(8, 8), subplot_kw={"projection": "polar"})
        for i, (subj, scores) in enumerate(list(scores_by_subject.items())[:6]):
            score_map = {s["factor_name"]: s["score"] for s in scores}
            vals = [score_map.get(f, 0) for f in all_factors] + [score_map.get(all_factors[0], 0)]
            ax.plot(angles, vals, color=self.colors[i % len(self.colors)],
                    linewidth=1.5, label=subj)
            ax.fill(angles, vals, color=self.colors[i % len(self.colors)], alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(all_factors, fontsize=self.theme.tick_size)
        ax.set_title(kwargs.get("title", "Confounding Factor Scores"), pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                  fontsize=self.theme.legend_fontsize)

        if output_dir:
            self.save(fig, "confounding_radar", output_dir)
        return fig

    # ── Consistency heatmap ───────────────────────────────────────

    def plot_consistency_heatmap(
        self,
        causality_results: dict[str, Any],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """
        Heatmap showing causality assessment consistency across subjects
        for each AE type matching IB risk categories.
        """
        consistency = causality_results.get("consistency_findings", [])
        fwd = causality_results.get("forward_findings", [])
        rev = causality_results.get("reverse_findings", [])

        all_f = fwd + rev
        if not all_f:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No causality data for heatmap", ha="center",
                    va="center", transform=ax.transAxes)
            return fig

        rows = []
        for f in all_f:
            cat = f.category.split("(")[-1].rstrip(")") if "(" in f.category else f.category
            direction = "forward" if "Forward" in f.category else "reverse"
            rows.append({"subject": f.subject_id, "category": cat, "direction": direction})

        df = pd.DataFrame(rows)
        score_map = {"forward": -1, "reverse": 1}
        df["value"] = df["direction"].map(score_map)
        pivot = df.pivot_table(index="subject", columns="category", values="value",
                               aggfunc="sum", fill_value=0)

        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="RdBu", zmid=0,
            ))
            fig.update_layout(title=kwargs.get("title", "Causality Assessment Consistency"))
            if output_dir:
                self.save_plotly(fig, "causality_consistency", output_dir)
            return fig

        import seaborn as sns
        fig, ax = self.create_figure(figsize=(max(6, len(pivot.columns) * 1.2),
                                              max(4, len(pivot) * 0.4)))
        sns.heatmap(pivot, ax=ax, cmap="RdBu_r", center=0, annot=True, fmt="g",
                    linewidths=0.5)
        ax.set_title(kwargs.get("title", "Causality Assessment Consistency"))

        if output_dir:
            self.save(fig, "causality_consistency", output_dir)
        return fig
