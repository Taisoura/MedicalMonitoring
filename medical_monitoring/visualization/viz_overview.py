"""
Module A visualizations: EDC data overview.

Charts:
    plot_consort         -- CONSORT-style subject disposition flow diagram
    plot_crf_heatmap     -- CRF completion rate heatmap (subject x form)
    plot_enrollment      -- Enrollment cumulative curve by site
    plot_quality_radar   -- Data-quality radar (missing / logic / terminology)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..parsers.edc_parser import EDCDataset
from .base import VizBase


class OverviewVisualizer(VizBase):
    """Visualization suite for EDC data overview (Module A)."""

    # ── CONSORT flow diagram ──────────────────────────────────────

    def plot_consort(
        self,
        dataset: EDCDataset,
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """CONSORT-style subject disposition flow diagram."""
        stages = self._extract_disposition(dataset)

        if interactive:
            return self._consort_plotly(stages, output_dir=output_dir, **kwargs)

        fig, ax = self.create_figure(figsize=(8, 10))
        box_props = dict(
            boxstyle="round,pad=0.4",
            facecolor=self.colors[1],
            edgecolor=self.colors[0],
            alpha=0.85,
        )

        y_positions = np.linspace(0.9, 0.1, len(stages))
        for i, (label, n) in enumerate(stages):
            ax.text(
                0.5, y_positions[i], f"{label}\n(n = {n})",
                transform=ax.transAxes, fontsize=self.theme.label_size,
                ha="center", va="center", bbox=box_props,
            )
            if i < len(stages) - 1:
                ax.annotate(
                    "", xy=(0.5, y_positions[i + 1] + 0.035),
                    xytext=(0.5, y_positions[i] - 0.035),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=self.colors[0], lw=1.5),
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title(
            kwargs.get("title", "Subject Disposition (CONSORT)"),
            fontsize=self.theme.title_size, fontweight=self.theme.font_weight,
        )

        if output_dir:
            self.save(fig, "consort_flow", output_dir)
        return fig

    def _consort_plotly(self, stages: list, **kwargs: Any) -> go.Figure:
        labels = [s[0] for s in stages]
        values = [s[1] for s in stages]
        source = list(range(len(stages) - 1))
        target = list(range(1, len(stages)))

        fig = self.create_plotly_figure()
        fig.add_trace(go.Sankey(
            node=dict(label=labels, color=self.colors[:len(labels)]),
            link=dict(source=source, target=target, value=values[:-1]),
        ))
        fig.update_layout(title_text=kwargs.get("title", "Subject Disposition"))
        if kwargs.get("output_dir"):
            self.save_plotly(fig, "consort_flow", kwargs["output_dir"])
        return fig

    @staticmethod
    def _extract_disposition(dataset: EDCDataset) -> list[tuple[str, int]]:
        n_total = dataset.n_subjects
        ds_df = dataset.get_table("DS")
        ie_df = dataset.get_table("IE")

        screened = n_total
        enrolled = n_total
        if ie_df is not None:
            eligible = ie_df[ie_df.get("field_0", pd.Series(dtype=str)).astype(str).str.contains("符合|合格|eligible", case=False, na=False)]
            enrolled = max(len(eligible), n_total)

        completed = n_total
        discontinued = 0
        if ds_df is not None:
            disc = ds_df[ds_df.get("field_0", pd.Series(dtype=str)).astype(str).str.contains("退出|脱落|终止|discontinu", case=False, na=False)]
            discontinued = len(disc)
            completed = n_total - discontinued

        return [
            ("Screened", screened),
            ("Enrolled / Randomized", enrolled),
            ("Completed", completed),
            ("Discontinued", discontinued),
        ]

    # ── CRF completion heatmap ────────────────────────────────────

    def plot_crf_heatmap(
        self,
        dataset: EDCDataset,
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Heatmap of CRF form completion (subjects x forms)."""
        matrix, forms, subjects = self._build_completion_matrix(dataset)

        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Heatmap(
                z=matrix, x=forms, y=subjects,
                colorscale="Blues", zmin=0, zmax=matrix.max() if matrix.size else 1,
                colorbar=dict(title="Records"),
            ))
            fig.update_layout(
                title=kwargs.get("title", "CRF Completion Heatmap"),
                xaxis_title="CRF Form", yaxis_title="Subject",
            )
            if output_dir:
                self.save_plotly(fig, "crf_heatmap", output_dir)
            return fig

        import seaborn as sns

        n_subj = len(subjects)
        h = max(4, n_subj * 0.25)
        fig, ax = self.create_figure(figsize=(max(8, len(forms) * 0.6), h))
        sns.heatmap(
            pd.DataFrame(matrix, index=subjects, columns=forms),
            ax=ax, cmap="Blues", annot=n_subj <= 30, fmt="g",
            linewidths=0.5, cbar_kws={"label": "Records"},
        )
        ax.set_title(
            kwargs.get("title", "CRF Completion Heatmap"),
            fontsize=self.theme.title_size,
        )
        ax.set_xlabel("CRF Form")
        ax.set_ylabel("Subject")

        if output_dir:
            self.save(fig, "crf_heatmap", output_dir)
        return fig

    @staticmethod
    def _build_completion_matrix(
        dataset: EDCDataset,
    ) -> tuple[np.ndarray, list[str], list[str]]:
        forms = sorted(dataset.tables.keys())
        subjects = sorted(dataset.subject_ids)
        matrix = np.zeros((len(subjects), len(forms)), dtype=int)

        subj_idx = {s: i for i, s in enumerate(subjects)}
        form_idx = {f: j for j, f in enumerate(forms)}

        for form_name, df in dataset.tables.items():
            j = form_idx.get(form_name)
            if j is None or "subject_id" not in df.columns:
                continue
            for sid in df["subject_id"].dropna().unique():
                sid_norm = str(sid).strip().upper()
                i = subj_idx.get(sid_norm)
                if i is not None:
                    matrix[i, j] = int(df[df["subject_id"] == sid].shape[0])

        return matrix, forms, subjects

    # ── Enrollment trend ──────────────────────────────────────────

    def plot_enrollment(
        self,
        dataset: EDCDataset,
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Cumulative enrollment trend by site."""
        ic_df = dataset.get_table("IC")
        if ic_df is None:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "IC table not available", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        date_col = None
        for col in ic_df.columns:
            if "date" in col.lower() or col == "field_1":
                date_col = col
                break
        if date_col is None:
            date_col = "field_1"

        ic_copy = ic_df.copy()
        ic_copy["_date"] = pd.to_datetime(ic_copy[date_col], errors="coerce")
        ic_copy = ic_copy.dropna(subset=["_date"]).sort_values("_date")

        if interactive:
            fig = self.create_plotly_figure()
            ic_copy["_cum"] = range(1, len(ic_copy) + 1)
            fig.add_trace(go.Scatter(
                x=ic_copy["_date"], y=ic_copy["_cum"],
                mode="lines+markers", name="Cumulative Enrollment",
                line=dict(color=self.colors[0]),
            ))
            fig.update_layout(title=kwargs.get("title", "Enrollment Trend"))
            if output_dir:
                self.save_plotly(fig, "enrollment_trend", output_dir)
            return fig

        fig, ax = self.create_figure()
        cum = np.arange(1, len(ic_copy) + 1)
        ax.plot(ic_copy["_date"], cum, "-o", color=self.colors[0], markersize=4)
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Subjects")
        ax.set_title(kwargs.get("title", "Enrollment Trend"))
        fig.autofmt_xdate()

        if output_dir:
            self.save(fig, "enrollment_trend", output_dir)
        return fig

    # ── Data quality radar ────────────────────────────────────────

    def plot_quality_radar(
        self,
        quality_metrics: dict[str, float],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """
        Radar chart of data quality dimensions.

        quality_metrics: e.g. {"Completeness": 0.95, "Date Logic": 0.88, ...}
        Values should be 0-1 (proportion OK).
        """
        labels = list(quality_metrics.keys())
        values = list(quality_metrics.values())

        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                fillcolor=self.colors[1],
                opacity=0.4,
                line=dict(color=self.colors[0]),
                name="Quality Score",
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=kwargs.get("title", "Data Quality Overview"),
            )
            if output_dir:
                self.save_plotly(fig, "quality_radar", output_dir)
            return fig

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles += angles[:1]

        fig, ax = self.create_figure(figsize=(7, 7), subplot_kw={"projection": "polar"})
        ax.fill(angles, values_plot, color=self.colors[1], alpha=0.3)
        ax.plot(angles, values_plot, color=self.colors[0], linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=self.theme.tick_size)
        ax.set_ylim(0, 1)
        ax.set_title(
            kwargs.get("title", "Data Quality Overview"),
            fontsize=self.theme.title_size, pad=20,
        )

        if output_dir:
            self.save(fig, "quality_radar", output_dir)
        return fig
