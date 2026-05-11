"""
Module C visualizations: Subject-level database.

Charts:
    plot_demographics_pyramid  -- Age-sex population pyramid
    plot_patient_profile       -- Single-subject longitudinal timeline
    plot_upset                 -- UpSet plot for comorbidity combinations
    plot_spaghetti             -- Score trajectory spaghetti + mean overlay
    plot_swimmer               -- Swimmer plot of subject events over time
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .base import VizBase


class SubjectVisualizer(VizBase):
    """Visualization suite for subject-level data (Module C)."""

    # ── Demographics pyramid ──────────────────────────────────────

    def plot_demographics_pyramid(
        self,
        subject_db: pd.DataFrame,
        *,
        age_col: str = "age",
        sex_col: str = "sex",
        bins: list[int] | None = None,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Population pyramid of age-sex distribution."""
        if bins is None:
            bins = [0, 30, 40, 50, 60, 70, 80, 100]

        df = subject_db[[age_col, sex_col]].dropna().copy()
        df["age_bin"] = pd.cut(df[age_col], bins=bins)
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

        male_mask = df[sex_col].astype(str).str.contains("男|[Mm]ale|M|1", na=False)
        male_counts = df[male_mask].groupby("age_bin", observed=False).size().reindex(
            pd.CategoricalIndex(pd.cut([], bins=bins).categories), fill_value=0
        ).values
        female_counts = df[~male_mask].groupby("age_bin", observed=False).size().reindex(
            pd.CategoricalIndex(pd.cut([], bins=bins).categories), fill_value=0
        ).values

        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Bar(
                y=labels, x=-male_counts.astype(float), name="Male",
                orientation="h", marker_color=self.colors[0],
            ))
            fig.add_trace(go.Bar(
                y=labels, x=female_counts.astype(float), name="Female",
                orientation="h", marker_color=self.colors[3],
            ))
            fig.update_layout(
                barmode="overlay",
                title=kwargs.get("title", "Age-Sex Distribution"),
                xaxis_title="Count", yaxis_title="Age Group",
            )
            if output_dir:
                self.save_plotly(fig, "demographics_pyramid", output_dir)
            return fig

        fig, ax = self.create_figure(figsize=(8, 6))
        y = np.arange(len(labels))
        ax.barh(y, -male_counts, height=0.7, color=self.colors[0], label="Male")
        ax.barh(y, female_counts, height=0.7, color=self.colors[3], label="Female")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Count")
        ax.set_title(kwargs.get("title", "Age-Sex Distribution"))
        ax.legend()
        ax.axvline(0, color="black", linewidth=0.8)
        max_val = max(male_counts.max(), female_counts.max()) + 2
        ax.set_xlim(-max_val, max_val)

        if output_dir:
            self.save(fig, "demographics_pyramid", output_dir)
        return fig

    # ── Patient profile timeline ──────────────────────────────────

    def plot_patient_profile(
        self,
        subject_id: str,
        events: pd.DataFrame,
        *,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """
        Patient profile timeline showing AE/CM/LB events.

        events columns: category (AE/CM/LB/Visit), name, start_date, end_date, grade
        """
        if events.empty:
            fig = self.create_plotly_figure()
            fig.add_annotation(text=f"No events for {subject_id}", showarrow=False)
            return fig

        cat_colors = {
            "AE": self.severity_colors.get("high", "#D55E00"),
            "CM": self.colors[1],
            "LB": self.colors[2],
            "Visit": self.colors[3],
        }

        fig = self.create_plotly_figure()
        categories = events["category"].unique()
        for i, cat in enumerate(categories):
            cat_df = events[events["category"] == cat]
            for _, row in cat_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[pd.Timedelta(days=max(1, (row.get("end_date", row["start_date"]) - row["start_date"]).days))
                       if pd.notna(row.get("end_date")) else pd.Timedelta(days=3)],
                    y=[f"{cat}: {row['name'][:30]}"],
                    base=[row["start_date"]],
                    orientation="h",
                    marker_color=cat_colors.get(cat, "#999"),
                    showlegend=False,
                    hovertext=f"{row.get('name', '')} (Grade {row.get('grade', 'N/A')})",
                ))

        fig.update_layout(
            title=kwargs.get("title", f"Patient Profile: {subject_id}"),
            xaxis_title="Date",
            barmode="stack",
            height=max(400, len(events) * 25),
        )
        if output_dir:
            self.save_plotly(fig, f"patient_profile_{subject_id}", output_dir)
        return fig

    # ── UpSet plot ────────────────────────────────────────────────

    def plot_upset(
        self,
        subject_db: pd.DataFrame,
        comorbidity_cols: list[str] | None = None,
        *,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> plt.Figure:
        """
        UpSet-style plot for comorbidity co-occurrence.

        Falls back to horizontal bar chart if the combination space is large.
        """
        if comorbidity_cols is None:
            comorbidity_cols = [
                c for c in subject_db.columns
                if subject_db[c].dropna().isin([0, 1]).all() and c not in
                ["death", "sex", "ce1", "ce2", "ce3"]
            ]

        if len(comorbidity_cols) < 2:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "Insufficient comorbidity columns", ha="center",
                    va="center", transform=ax.transAxes)
            return fig

        bool_df = subject_db[comorbidity_cols].fillna(0).astype(int)
        combos = bool_df.apply(lambda row: tuple(row), axis=1)
        combo_counts = combos.value_counts().head(15)

        fig, (ax_bar, ax_dots) = self.create_figure(
            nrows=2, ncols=1, figsize=(max(8, len(combo_counts) * 0.8), 8),
            gridspec_kw={"height_ratios": [2, 1]},
        )

        x = np.arange(len(combo_counts))
        ax_bar.bar(x, combo_counts.values, color=self.colors[0], width=0.6)
        ax_bar.set_ylabel("Intersection Size")
        ax_bar.set_xticks([])
        ax_bar.set_title(kwargs.get("title", "Comorbidity Combinations"))

        for i, combo in enumerate(combo_counts.index):
            for j, (col_name, val) in enumerate(zip(comorbidity_cols, combo)):
                color = self.colors[0] if val else "#E0E0E0"
                ax_dots.scatter(i, j, s=80, color=color, zorder=3)
            active = [j for j, v in enumerate(combo) if v]
            if len(active) > 1:
                ax_dots.plot([i] * len(active), active, color=self.colors[0],
                             linewidth=2, zorder=2)

        ax_dots.set_yticks(range(len(comorbidity_cols)))
        ax_dots.set_yticklabels(comorbidity_cols, fontsize=self.theme.tick_size)
        ax_dots.set_xticks([])
        ax_dots.invert_yaxis()
        ax_dots.set_xlim(-0.5, len(combo_counts) - 0.5)

        fig.tight_layout()
        if output_dir:
            self.save(fig, "upset_comorbidity", output_dir)
        return fig

    # ── Spaghetti + mean trajectory ───────────────────────────────

    def plot_spaghetti(
        self,
        subject_db: pd.DataFrame,
        score_cols: list[str],
        *,
        score_name: str = "Score",
        highlight_subjects: list[str] | None = None,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Spaghetti plot of individual trajectories with mean overlay."""
        visits = list(range(len(score_cols)))

        if interactive:
            fig = self.create_plotly_figure()
            for _, row in subject_db.iterrows():
                vals = [row.get(c) for c in score_cols]
                sid = str(row.get("subject_id", ""))
                is_hl = highlight_subjects and sid in highlight_subjects
                fig.add_trace(go.Scatter(
                    x=visits, y=vals, mode="lines",
                    line=dict(
                        color=self.severity_colors["high"] if is_hl else "#CCCCCC",
                        width=2 if is_hl else 0.8,
                    ),
                    name=sid if is_hl else None,
                    showlegend=is_hl,
                    hovertext=sid,
                ))
            means = subject_db[score_cols].mean()
            fig.add_trace(go.Scatter(
                x=visits, y=means.values, mode="lines+markers",
                line=dict(color=self.colors[0], width=3),
                name="Mean",
            ))
            fig.update_layout(
                title=kwargs.get("title", f"{score_name} Trajectories"),
                xaxis=dict(tickvals=visits, ticktext=score_cols),
                yaxis_title=score_name,
            )
            if output_dir:
                self.save_plotly(fig, f"spaghetti_{score_name.lower()}", output_dir)
            return fig

        fig, ax = self.create_figure()
        for _, row in subject_db.iterrows():
            vals = [row.get(c) for c in score_cols]
            sid = str(row.get("subject_id", ""))
            is_hl = highlight_subjects and sid in highlight_subjects
            ax.plot(visits, vals, alpha=0.8 if is_hl else 0.25,
                    color=self.severity_colors["high"] if is_hl else "#AAAAAA",
                    linewidth=2 if is_hl else 0.6)

        means = subject_db[score_cols].mean()
        ax.plot(visits, means.values, "-o", color=self.colors[0], linewidth=3,
                label="Mean", zorder=10)
        ax.set_xticks(visits)
        ax.set_xticklabels(score_cols, fontsize=self.theme.tick_size)
        ax.set_ylabel(score_name)
        ax.set_title(kwargs.get("title", f"{score_name} Trajectories"))
        ax.legend()

        if output_dir:
            self.save(fig, f"spaghetti_{score_name.lower()}", output_dir)
        return fig

    # ── Swimmer plot ──────────────────────────────────────────────

    def plot_swimmer(
        self,
        subject_db: pd.DataFrame,
        *,
        duration_col: str | None = None,
        event_cols: list[str] | None = None,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """
        Swimmer plot of per-subject study duration with event markers.

        If duration_col is not available, uses a simple index-based layout.
        """
        df = subject_db.copy()
        if "subject_id" not in df.columns:
            df["subject_id"] = [f"S{i}" for i in range(len(df))]

        if duration_col and duration_col in df.columns:
            df = df.sort_values(duration_col, ascending=True)
            durations = df[duration_col].fillna(0).values
        else:
            durations = np.ones(len(df)) * 90  # default 90 days

        subjects = df["subject_id"].astype(str).values
        y = np.arange(len(subjects))

        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Bar(
                y=subjects, x=durations, orientation="h",
                marker_color=self.colors[1], name="Study Duration",
            ))
            if event_cols:
                for j, ev_col in enumerate(event_cols):
                    if ev_col in df.columns:
                        mask = df[ev_col].fillna(0).astype(bool)
                        fig.add_trace(go.Scatter(
                            x=durations[mask], y=subjects[mask],
                            mode="markers",
                            marker=dict(symbol="diamond", size=10,
                                        color=self.colors[min(j + 2, len(self.colors) - 1)]),
                            name=ev_col,
                        ))
            fig.update_layout(
                title=kwargs.get("title", "Swimmer Plot"),
                xaxis_title="Days on Study", yaxis_title="Subject",
            )
            if output_dir:
                self.save_plotly(fig, "swimmer_plot", output_dir)
            return fig

        fig, ax = self.create_figure(figsize=(10, max(4, len(subjects) * 0.25)))
        ax.barh(y, durations, height=0.6, color=self.colors[1], label="Duration")

        if event_cols:
            for j, ev_col in enumerate(event_cols):
                if ev_col in df.columns:
                    mask = df[ev_col].fillna(0).astype(bool).values
                    ci = min(j + 2, len(self.colors) - 1)
                    ax.scatter(
                        durations[mask], y[mask], s=60, zorder=5,
                        color=self.colors[ci], marker="D", label=ev_col,
                    )

        ax.set_yticks(y)
        ax.set_yticklabels(subjects, fontsize=max(5, self.theme.tick_size - 2))
        ax.set_xlabel("Days on Study")
        ax.set_title(kwargs.get("title", "Swimmer Plot"))
        ax.legend(loc="lower right", fontsize=self.theme.legend_fontsize)

        if output_dir:
            self.save(fig, "swimmer_plot", output_dir)
        return fig
