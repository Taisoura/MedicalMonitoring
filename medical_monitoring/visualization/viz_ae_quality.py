"""
Module F1 visualizations: AE quality audit results.

Charts:
    plot_quality_radar      -- 10-dimension AE quality radar / polar
    plot_naming_sunburst    -- Naming variant sunburst (standard → variants)
    plot_action_gantt       -- Action plan timeline with P1/P2/P3 priority
    plot_cross_ref_matrix   -- Cross-reference impact matrix
    plot_severity_bar       -- AE severity grade distribution stacked bar
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ..audit.ae_quality import ActionItem, NamingVariant
from ..utils.helpers import Finding
from .base import VizBase


class AEQualityVisualizer(VizBase):
    """Visualization suite for AE quality audit (Module F1)."""

    # ── 10-dimension quality radar ────────────────────────────────

    def plot_quality_radar(
        self,
        ae_results: dict[str, Any],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Radar chart of AE quality across 10 audit dimensions."""
        dimension_labels = [
            "Naming", "Dates", "Outcomes", "Severity",
            "SAE Gaps", "Drug Action", "Death", "NCS",
            "Cross-Ref", "Action Plan",
        ]
        dimension_keys = [
            "1_naming", "2_dates", "3_outcomes", "4_severity",
            "5_sae_gaps", "6_drug_action", "7_death", "8_ncs",
            "9_cross_ref", "10_action_plan",
        ]

        total_ae = ae_results.get("summary", {}).get("total_ae_records", 1)
        scores = []
        for key in dimension_keys:
            findings = ae_results.get(key, [])
            if isinstance(findings, list):
                n_issues = len(findings)
            else:
                n_issues = 0
            score = max(0, 1 - n_issues / max(total_ae, 1))
            scores.append(round(score, 3))

        if interactive:
            fig = self.create_plotly_figure()
            fig.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],
                theta=dimension_labels + [dimension_labels[0]],
                fill="toself",
                fillcolor=self.colors[1],
                opacity=0.4,
                line=dict(color=self.colors[0]),
                name="Quality Score",
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=kwargs.get("title", "AE Quality: 10-Dimension Radar"),
            )
            if output_dir:
                self.save_plotly(fig, "ae_quality_radar", output_dir)
            return fig

        angles = np.linspace(0, 2 * np.pi, len(dimension_labels), endpoint=False).tolist()
        vals = scores + [scores[0]]
        angles += angles[:1]

        fig, ax = self.create_figure(figsize=(8, 8), subplot_kw={"projection": "polar"})
        ax.fill(angles, vals, color=self.colors[1], alpha=0.3)
        ax.plot(angles, vals, color=self.colors[0], linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimension_labels, fontsize=self.theme.tick_size)
        ax.set_ylim(0, 1)
        ax.set_title(kwargs.get("title", "AE Quality: 10-Dimension Radar"), pad=20)

        if output_dir:
            self.save(fig, "ae_quality_radar", output_dir)
        return fig

    # ── Naming variant sunburst ───────────────────────────────────

    def plot_naming_sunburst(
        self,
        naming_variants: list[NamingVariant] | list[dict],
        *,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Sunburst: centre = standard term, outer ring = variants."""
        ids = ["AE Terms"]
        labels = ["AE Terms"]
        parents = [""]
        values = [0]

        for nv in naming_variants:
            if isinstance(nv, dict):
                std = nv.get("standard_term", "?")
                variants = nv.get("variants", [])
                affected = nv.get("affected_subjects", [])
            else:
                std = nv.standard_term
                variants = nv.variants
                affected = nv.affected_subjects

            ids.append(std)
            labels.append(std)
            parents.append("AE Terms")
            values.append(len(affected))

            for v in variants[:10]:
                uid = f"{std}__{v}"
                ids.append(uid)
                labels.append(v)
                parents.append(std)
                values.append(max(1, len(affected) // max(len(variants), 1)))

        fig = self.create_plotly_figure()
        fig.add_trace(go.Sunburst(
            ids=ids, labels=labels, parents=parents, values=values,
            branchvalues="total",
            marker=dict(colors=self.theme.sequential[:len(ids)]),
        ))
        fig.update_layout(title=kwargs.get("title", "AE Naming Variant Sunburst"))
        if output_dir:
            self.save_plotly(fig, "naming_sunburst", output_dir)
        return fig

    # ── Action plan Gantt ─────────────────────────────────────────

    def plot_action_gantt(
        self,
        action_items: list[ActionItem] | list[dict],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Timeline of action items coloured by priority."""
        if not action_items:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No action items", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        items: list[dict] = []
        for ai in action_items:
            d = ai if isinstance(ai, dict) else ai.__dict__
            items.append(d)

        priority_order = {"urgent": 0, "high": 1, "medium": 2}
        items.sort(key=lambda x: priority_order.get(x.get("priority", "medium"), 9))
        deadline_days = {"24-48h": 2, "This week": 7, "Two weeks": 14}

        if interactive:
            fig = self.create_plotly_figure()
            for i, item in enumerate(items):
                prio = item.get("priority", "medium")
                days = deadline_days.get(item.get("deadline", "Two weeks"), 14)
                fig.add_trace(go.Bar(
                    y=[f"{item.get('item_id', '')} - {item.get('problem', '')[:40]}"],
                    x=[days], orientation="h",
                    marker_color=self.color_for_severity(prio),
                    name=prio.capitalize() if i < 3 else None,
                    showlegend=i < 3,
                ))
            fig.update_layout(
                title=kwargs.get("title", "Action Plan Timeline"),
                xaxis_title="Days to Deadline",
                barmode="stack",
            )
            if output_dir:
                self.save_plotly(fig, "action_gantt", output_dir)
            return fig

        fig, ax = self.create_figure(figsize=(10, max(3, len(items) * 0.35)))
        y = np.arange(len(items))
        labels = []
        for i, item in enumerate(items):
            prio = item.get("priority", "medium")
            days = deadline_days.get(item.get("deadline", "Two weeks"), 14)
            ax.barh(i, days, color=self.color_for_severity(prio), height=0.6)
            labels.append(f"{item.get('item_id', '')} {item.get('problem', '')[:45]}")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=max(5, self.theme.tick_size - 2))
        ax.set_xlabel("Days to Deadline")
        ax.set_title(kwargs.get("title", "Action Plan Timeline"))

        if output_dir:
            self.save(fig, "action_gantt", output_dir)
        return fig

    # ── Cross-reference matrix ────────────────────────────────────

    def plot_cross_ref_matrix(
        self,
        cross_refs: list[dict],
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Cross-reference impact matrix between AE audit and other modules."""
        if not cross_refs:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No cross-reference data", ha="center", va="center",
                    transform=ax.transAxes)
            return fig

        audits = [cr.get("related_audit", "") for cr in cross_refs]
        findings = [cr.get("finding", "")[:40] for cr in cross_refs]
        impacts = [cr.get("ae_quality_impact", "")[:40] for cr in cross_refs]
        actions = [cr.get("priority_action", "")[:40] for cr in cross_refs]

        if interactive:
            header_vals = ["Related Audit", "Finding", "AE Impact", "Priority Action"]
            cell_vals = [audits, findings, impacts, actions]
            fig = self.create_plotly_figure()
            fig.add_trace(go.Table(
                header=dict(values=header_vals,
                            fill_color=self.colors[0],
                            font=dict(color="white")),
                cells=dict(values=cell_vals,
                           fill_color=self.colors[1],
                           font=dict(color="black")),
            ))
            fig.update_layout(title=kwargs.get("title", "Cross-Reference Impact Matrix"))
            if output_dir:
                self.save_plotly(fig, "cross_ref_matrix", output_dir)
            return fig

        fig, ax = self.create_figure(figsize=(14, max(3, len(cross_refs) * 0.8)))
        ax.axis("off")
        table_data = [[a, f, im, ac] for a, f, im, ac in zip(audits, findings, impacts, actions)]
        table = ax.table(
            cellText=table_data,
            colLabels=["Related Audit", "Finding", "AE Impact", "Priority Action"],
            loc="center", cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(self.theme.tick_size)
        table.scale(1.0, 1.5)
        ax.set_title(kwargs.get("title", "Cross-Reference Impact Matrix"),
                     fontsize=self.theme.title_size)

        if output_dir:
            self.save(fig, "cross_ref_matrix", output_dir)
        return fig

    # ── Severity distribution stacked bar ─────────────────────────

    def plot_severity_bar(
        self,
        ae_df: pd.DataFrame | None = None,
        ae_results: dict[str, Any] | None = None,
        *,
        output_dir: Path | None = None,
        interactive: bool = False,
        **kwargs: Any,
    ) -> plt.Figure | go.Figure:
        """Stacked bar of AE severity grade distribution by SOC or PT."""
        all_findings: list[Finding] = []
        if ae_results:
            for key in ["1_naming", "2_dates", "3_outcomes", "4_severity",
                        "5_sae_gaps", "6_drug_action", "7_death"]:
                val = ae_results.get(key, [])
                if isinstance(val, list):
                    all_findings.extend(val)

        if not all_findings:
            fig, ax = self.create_figure()
            ax.text(0.5, 0.5, "No findings for severity chart", ha="center",
                    va="center", transform=ax.transAxes)
            return fig

        sev_counter: dict[str, Counter] = {}
        for f in all_findings:
            cat = f.category
            if cat not in sev_counter:
                sev_counter[cat] = Counter()
            sev_counter[cat][f.severity] += 1

        categories = sorted(sev_counter.keys())

        if interactive:
            fig = self.create_plotly_figure()
            for sev in ["urgent", "high", "medium", "low"]:
                fig.add_trace(go.Bar(
                    x=categories,
                    y=[sev_counter[c].get(sev, 0) for c in categories],
                    name=sev.capitalize(),
                    marker_color=self.color_for_severity(sev),
                ))
            fig.update_layout(
                barmode="stack",
                title=kwargs.get("title", "AE Findings by Category & Severity"),
                yaxis_title="Count",
            )
            if output_dir:
                self.save_plotly(fig, "ae_severity_bar", output_dir)
            return fig

        fig, ax = self.create_figure(figsize=(max(8, len(categories) * 1.0), 6))
        x = np.arange(len(categories))
        bottom = np.zeros(len(categories))
        for sev in ["low", "medium", "high", "urgent"]:
            vals = np.array([sev_counter[c].get(sev, 0) for c in categories], dtype=float)
            ax.bar(x, vals, bottom=bottom, color=self.color_for_severity(sev),
                   label=sev.capitalize(), width=0.6)
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=30, ha="right",
                           fontsize=self.theme.tick_size)
        ax.set_ylabel("Finding Count")
        ax.set_title(kwargs.get("title", "AE Findings by Category & Severity"))
        ax.legend()

        if output_dir:
            self.save(fig, "ae_severity_bar", output_dir)
        return fig
