"""
Module D visualizations: Statistical analysis toolbox.

Charts:
    plot_forest        -- Forest plot of OR / HR with 95% CI
    plot_kaplan_meier  -- Kaplan-Meier survival curves with risk table
    plot_shift_table   -- Shift table heatmap (baseline vs post-baseline)
    plot_volcano       -- PRR volcano plot for safety signal detection
    plot_roc           -- ROC curve with AUC annotation
    plot_bland_altman  -- Bland-Altman agreement plot
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..analysis.statistics import ConsistencyFinding, ORResult
from .base import VizBase


class StatisticsVisualizer(VizBase):
    """Visualization suite for statistical results (Module D)."""

    # ── Forest Plot ───────────────────────────────────────────────

    def plot_forest(
        self,
        or_results: list[ORResult] | list[dict],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Forest plot of OR values with 95% CI and reference line at OR=1."""
        data = []
        for r in or_results:
            d = r if isinstance(r, dict) else r.__dict__ if hasattr(r, "__dict__") else {}
            if not np.isfinite(d.get("odds_ratio", float("nan"))):
                continue
            data.append(d)

        if not data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No valid OR data", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        data.sort(key=lambda x: x.get("odds_ratio", 1))
        labels = [d.get("predictor", "") for d in data]
        ors = [d.get("odds_ratio", 1) for d in data]
        ci_low = [d.get("ci_lower", 0) for d in data]
        ci_high = [d.get("ci_upper", 0) for d in data]
        pvals = [d.get("p_value", 1) for d in data]

        if interactive:
            fig = self.create_plotly_figure()
            for i, (lab, orv, lo, hi, pv) in enumerate(
                zip(labels, ors, ci_low, ci_high, pvals)
            ):
                fig.add_trace(go.Scatter(
                    x=[lo, hi], y=[lab, lab], mode="lines",
                    line=dict(color=self.colors[0], width=2),
                    showlegend=False,
                ))
                sig = pv < 0.05
                fig.add_trace(go.Scatter(
                    x=[orv], y=[lab], mode="markers",
                    marker=dict(
                        size=10, color=self.colors[0] if sig else "#AAAAAA",
                        symbol="diamond",
                    ),
                    showlegend=False,
                    hovertext=f"OR={orv:.2f} ({lo:.2f}-{hi:.2f}), p={pv:.4f}",
                ))

            fig.add_vline(x=1, line_dash="dash", line_color="grey")
            fig.update_layout(
                title=kwargs.get("title", "Forest Plot (Odds Ratios)"),
                xaxis_title="Odds Ratio (95% CI)",
                xaxis_type="log",
            )
            if output_dir:
                self.save_plotly(fig, "forest_plot", output_dir)
            return fig

        fig, ax = self.create_figure(figsize=(10, max(4, len(data) * 0.5)))
        y = np.arange(len(data))

        for i, (orv, lo, hi, pv) in enumerate(zip(ors, ci_low, ci_high, pvals)):
            colour = self.colors[0] if pv < 0.05 else "#AAAAAA"
            ax.plot([lo, hi], [i, i], color=colour, linewidth=2, solid_capstyle="round")
            ax.plot(orv, i, "D", color=colour, markersize=8)

        ax.axvline(1, color="grey", linestyle="--", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=self.theme.tick_size)
        ax.set_xlabel("Odds Ratio (95% CI)")
        ax.set_xscale("log")
        ax.set_title(kwargs.get("title", "Forest Plot (Odds Ratios)"))

        for i, (orv, lo, hi, pv) in enumerate(zip(ors, ci_low, ci_high, pvals)):
            ax.text(
                max(ci_high) * 1.2, i,
                f"{orv:.2f} ({lo:.2f}-{hi:.2f}) p={pv:.3f}",
                va="center", fontsize=self.theme.annotation_size,
            )

        if output_dir:
            self.save(fig, "forest_plot", output_dir)
        return fig

    # ── Kaplan-Meier ──────────────────────────────────────────────

    def plot_kaplan_meier(
        self,
        time: np.ndarray | pd.Series,
        event: np.ndarray | pd.Series,
        group: np.ndarray | pd.Series | None = None,
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Kaplan-Meier survival curve with optional grouping and risk table."""
        time = np.asarray(time, dtype=float)
        event = np.asarray(event, dtype=int)

        if group is None:
            groups = {"All": np.ones(len(time), dtype=bool)}
        else:
            group = np.asarray(group)
            groups = {g: group == g for g in np.unique(group)}

        if interactive:
            fig = self.create_plotly_figure()
            for i, (gname, mask) in enumerate(groups.items()):
                t, s = self._km_curve(time[mask], event[mask])
                ci = i % len(self.colors)
                fig.add_trace(go.Scatter(
                    x=t, y=s, mode="lines", name=gname,
                    line=dict(color=self.colors[ci], width=2),
                ))
            fig.update_layout(
                title=kwargs.get("title", "Kaplan-Meier Survival"),
                xaxis_title=kwargs.get("xaxis_title", "Time"),
                yaxis_title="Survival Probability",
                yaxis_range=[0, 1.05],
            )
            if output_dir:
                self.save_plotly(fig, "kaplan_meier", output_dir)
            return fig

        fig, ax = self.create_figure()
        for i, (gname, mask) in enumerate(groups.items()):
            t, s = self._km_curve(time[mask], event[mask])
            ci = i % len(self.colors)
            ax.step(t, s, where="post", color=self.colors[ci], linewidth=2, label=gname)

        ax.set_ylim(0, 1.05)
        ax.set_xlabel(kwargs.get("xaxis_title", "Time"))
        ax.set_ylabel("Survival Probability")
        ax.set_title(kwargs.get("title", "Kaplan-Meier Survival"))
        ax.legend()

        if output_dir:
            self.save(fig, "kaplan_meier", output_dir)
        return fig

    @staticmethod
    def _km_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simple Kaplan-Meier estimator (no external dependency)."""
        order = np.argsort(time)
        t_sorted = time[order]
        e_sorted = event[order]
        unique_times = np.unique(t_sorted)

        surv = 1.0
        times = [0.0]
        survival = [1.0]
        n_at_risk = len(t_sorted)

        for ut in unique_times:
            mask = t_sorted == ut
            d = e_sorted[mask].sum()
            if n_at_risk > 0 and d > 0:
                surv *= 1 - d / n_at_risk
            n_at_risk -= mask.sum()
            times.append(ut)
            survival.append(surv)

        return np.array(times), np.array(survival)

    # ── Shift Table Heatmap ───────────────────────────────────────

    def plot_shift_table(
        self,
        shift_df: pd.DataFrame,
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Heatmap of baseline-to-post-baseline shift table."""
        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Heatmap(
                z=shift_df.values,
                x=[str(c) for c in shift_df.columns],
                y=[str(r) for r in shift_df.index],
                colorscale="Blues",
                text=shift_df.values, texttemplate="%{text}",
            ))
            fig.update_layout(
                title=kwargs.get("title", "Laboratory Shift Table"),
                xaxis_title="Post-Baseline", yaxis_title="Baseline",
            )
            if output_dir:
                self.save_plotly(fig, "shift_table", output_dir)
            return fig

        import seaborn as sns
        fig, ax = self.create_figure(figsize=(8, 6))
        sns.heatmap(shift_df, ax=ax, annot=True, fmt="g", cmap="Blues",
                    linewidths=0.5, cbar_kws={"label": "Count"})
        ax.set_title(kwargs.get("title", "Laboratory Shift Table"))
        ax.set_xlabel("Post-Baseline")
        ax.set_ylabel("Baseline")

        if output_dir:
            self.save(fig, "shift_table", output_dir)
        return fig

    # ── PRR Volcano Plot ──────────────────────────────────────────

    def plot_volcano(
        self,
        prr_data: list[dict[str, Any]],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """
        Volcano plot for safety signal detection.

        prr_data: list of dicts with keys: ae_term, prr, chi2, signal, n
        """
        if not prr_data:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No PRR data", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        terms = [d["ae_term"] for d in prr_data]
        log_prr = [np.log2(max(d.get("prr", 1), 0.01)) for d in prr_data]
        neg_log_chi2 = [d.get("chi2", 0) for d in prr_data]
        signals = [d.get("signal", False) for d in prr_data]

        if interactive:
            fig = self.create_plotly_figure()
            colors = [self.severity_colors["urgent"] if s else "#AAAAAA" for s in signals]
            fig.add_trace(go.Scatter(
                x=log_prr, y=neg_log_chi2, mode="markers+text",
                marker=dict(size=10, color=colors),
                text=[t if s else "" for t, s in zip(terms, signals)],
                textposition="top center",
                hovertext=terms,
            ))
            fig.add_vline(x=np.log2(2), line_dash="dash", line_color="grey")
            fig.add_hline(y=4, line_dash="dash", line_color="grey")
            fig.update_layout(
                title=kwargs.get("title", "Safety Signal Volcano Plot"),
                xaxis_title="log2(PRR)", yaxis_title="Chi-squared",
            )
            if output_dir:
                self.save_plotly(fig, "volcano_prr", output_dir)
            return fig

        fig, ax = self.create_figure()
        for i, (lp, nc, s, t) in enumerate(zip(log_prr, neg_log_chi2, signals, terms)):
            c = self.severity_colors["urgent"] if s else "#AAAAAA"
            ax.scatter(lp, nc, c=c, s=60, zorder=3)
            if s:
                ax.annotate(t, (lp, nc), fontsize=self.theme.annotation_size,
                            ha="center", va="bottom")

        ax.axvline(np.log2(2), color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(4, color="grey", linestyle="--", linewidth=0.8)
        ax.set_xlabel("log2(PRR)")
        ax.set_ylabel("Chi-squared")
        ax.set_title(kwargs.get("title", "Safety Signal Volcano Plot"))

        if output_dir:
            self.save(fig, "volcano_prr", output_dir)
        return fig

    # ── ROC Curve ─────────────────────────────────────────────────

    def plot_roc(
        self,
        y_true: np.ndarray | pd.Series,
        y_score: np.ndarray | pd.Series,
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """ROC curve with AUC and optimal cut-point annotation."""
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score, dtype=float)

        fpr, tpr, thresholds = self._roc_curve(y_true, y_score)
        auc = np.trapz(tpr, fpr)

        j_index = tpr - fpr
        best_idx = np.argmax(j_index)
        best_thr = thresholds[best_idx] if best_idx < len(thresholds) else None

        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                line=dict(color=self.colors[0], width=2),
                name=f"ROC (AUC = {auc:.3f})",
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines",
                line=dict(color="grey", dash="dash"), showlegend=False,
            ))
            if best_thr is not None:
                fig.add_trace(go.Scatter(
                    x=[fpr[best_idx]], y=[tpr[best_idx]], mode="markers",
                    marker=dict(size=12, color=self.severity_colors["high"]),
                    name=f"Optimal (thr={best_thr:.2f})",
                ))
            fig.update_layout(
                title=kwargs.get("title", f"ROC Curve (AUC = {auc:.3f})"),
                xaxis_title="1 - Specificity", yaxis_title="Sensitivity",
            )
            if output_dir:
                self.save_plotly(fig, "roc_curve", output_dir)
            return fig

        fig, ax = self.create_figure(figsize=(7, 7))
        ax.plot(fpr, tpr, color=self.colors[0], linewidth=2,
                label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=0.8)
        if best_thr is not None:
            ax.plot(fpr[best_idx], tpr[best_idx], "o",
                    color=self.severity_colors["high"], markersize=10,
                    label=f"Optimal (thr={best_thr:.2f})")
        ax.set_xlabel("1 - Specificity")
        ax.set_ylabel("Sensitivity")
        ax.set_title(kwargs.get("title", f"ROC Curve (AUC = {auc:.3f})"))
        ax.legend()

        if output_dir:
            self.save(fig, "roc_curve", output_dir)
        return fig

    @staticmethod
    def _roc_curve(
        y_true: np.ndarray, y_score: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple ROC curve computation (no sklearn dependency)."""
        desc_score_indices = np.argsort(y_score)[::-1]
        y_score_sorted = y_score[desc_score_indices]
        y_true_sorted = y_true[desc_score_indices]

        thresholds = np.unique(y_score_sorted)[::-1]
        tpr_list = []
        fpr_list = []
        p = y_true.sum()
        n = len(y_true) - p

        for thr in thresholds:
            predicted = (y_score >= thr).astype(int)
            tp = ((predicted == 1) & (y_true == 1)).sum()
            fp = ((predicted == 1) & (y_true == 0)).sum()
            tpr_list.append(tp / max(p, 1))
            fpr_list.append(fp / max(n, 1))

        fpr_arr = np.array([0.0] + fpr_list + [1.0])
        tpr_arr = np.array([0.0] + tpr_list + [1.0])
        return fpr_arr, tpr_arr, thresholds

    # ── Bland-Altman ──────────────────────────────────────────────

    def plot_bland_altman(
        self,
        measure1: np.ndarray | pd.Series,
        measure2: np.ndarray | pd.Series,
        *,
        labels: tuple[str, str] = ("Measure 1", "Measure 2"),
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Bland-Altman agreement plot (difference vs mean)."""
        m1 = np.asarray(measure1, dtype=float)
        m2 = np.asarray(measure2, dtype=float)
        valid = np.isfinite(m1) & np.isfinite(m2)
        m1, m2 = m1[valid], m2[valid]

        mean_vals = (m1 + m2) / 2
        diff_vals = m1 - m2
        mean_diff = np.mean(diff_vals)
        sd_diff = np.std(diff_vals, ddof=1)
        upper_loa = mean_diff + 1.96 * sd_diff
        lower_loa = mean_diff - 1.96 * sd_diff

        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Scatter(
                x=mean_vals, y=diff_vals, mode="markers",
                marker=dict(color=self.colors[0], size=6),
                name="Data",
            ))
            fig.add_hline(y=mean_diff, line_dash="solid", line_color=self.colors[0],
                          annotation_text=f"Mean = {mean_diff:.2f}")
            fig.add_hline(y=upper_loa, line_dash="dash",
                          line_color=self.severity_colors["high"],
                          annotation_text=f"+1.96SD = {upper_loa:.2f}")
            fig.add_hline(y=lower_loa, line_dash="dash",
                          line_color=self.severity_colors["high"],
                          annotation_text=f"-1.96SD = {lower_loa:.2f}")
            fig.update_layout(
                title=kwargs.get("title", f"Bland-Altman: {labels[0]} vs {labels[1]}"),
                xaxis_title=f"Mean of {labels[0]} & {labels[1]}",
                yaxis_title=f"Difference ({labels[0]} - {labels[1]})",
            )
            if output_dir:
                self.save_plotly(fig, "bland_altman", output_dir)
            return fig

        fig, ax = self.create_figure()
        ax.scatter(mean_vals, diff_vals, c=self.colors[0], s=30, alpha=0.7)
        ax.axhline(mean_diff, color=self.colors[0], linewidth=1, label=f"Mean = {mean_diff:.2f}")
        ax.axhline(upper_loa, color=self.severity_colors["high"], linestyle="--",
                    label=f"+1.96SD = {upper_loa:.2f}")
        ax.axhline(lower_loa, color=self.severity_colors["high"], linestyle="--",
                    label=f"-1.96SD = {lower_loa:.2f}")
        ax.set_xlabel(f"Mean of {labels[0]} & {labels[1]}")
        ax.set_ylabel(f"Difference ({labels[0]} - {labels[1]})")
        ax.set_title(kwargs.get("title", f"Bland-Altman: {labels[0]} vs {labels[1]}"))
        ax.legend(fontsize=self.theme.legend_fontsize)

        if output_dir:
            self.save(fig, "bland_altman", output_dir)
        return fig
