"""
VizBase -- shared base class for all visualization modules.

Provides:
    - Theme application to matplotlib figures
    - Unified save / export workflow (PNG, SVG, PDF)
    - Plotly figure creation with theme-aware template
    - Plotly-to-HTML export
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

from .themes import ThemeConfig, get_theme

matplotlib.use("Agg")


class VizBase:
    """Base class for themed, export-capable visualizations."""

    def __init__(self, theme: str | ThemeConfig = "fda_regulatory"):
        if isinstance(theme, str):
            self.theme = get_theme(theme)
        else:
            self.theme = theme

    # ── matplotlib helpers ─────────────────────────────────────────

    def _apply_theme(self, ax: plt.Axes | None = None) -> None:
        """Apply the current theme to matplotlib's rcParams and an optional Axes."""
        t = self.theme
        plt.rcParams.update({
            "font.family": t.font_family,
            "font.size": t.label_size,
            "axes.titlesize": t.title_size,
            "axes.labelsize": t.label_size,
            "xtick.labelsize": t.tick_size,
            "ytick.labelsize": t.tick_size,
            "legend.fontsize": t.legend_fontsize,
            "figure.dpi": t.dpi,
            "figure.facecolor": t.background_color,
            "axes.facecolor": t.background_color,
            "axes.grid": True,
            "grid.alpha": t.grid_alpha,
            "grid.linestyle": t.grid_style,
            "grid.color": t.grid_color,
        })
        if ax is not None:
            for spine in ax.spines.values():
                spine.set_visible(t.spine_visible)

    def create_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: tuple[float, float] | None = None,
        **kwargs: Any,
    ) -> tuple[plt.Figure, Any]:
        """Create a themed matplotlib figure + axes."""
        self._apply_theme()
        if figsize is None:
            figsize = (self.theme.figure_width, self.theme.figure_height)
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

        if isinstance(axes, plt.Axes):
            self._apply_theme(axes)
        else:
            import numpy as np
            for ax in np.asarray(axes).flat:
                self._apply_theme(ax)

        fig.patch.set_facecolor(self.theme.background_color)
        return fig, axes

    def save(
        self,
        fig: plt.Figure,
        name: str,
        output_dir: Path | str | None = None,
        formats: tuple[str, ...] = ("png",),
        close: bool = True,
    ) -> list[Path]:
        """Save a matplotlib figure in one or more formats."""
        if output_dir is None:
            output_dir = Path(".")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        for fmt in formats:
            path = output_dir / f"{name}.{fmt}"
            fig.savefig(
                str(path),
                format=fmt,
                dpi=self.theme.dpi,
                bbox_inches="tight",
                facecolor=fig.get_facecolor(),
            )
            saved.append(path)

        if close:
            plt.close(fig)
        return saved

    # ── Plotly helpers ─────────────────────────────────────────────

    def create_plotly_figure(self, **kwargs: Any) -> go.Figure:
        """Create a plotly Figure with theme-appropriate template."""
        fig = go.Figure(**kwargs)
        fig.update_layout(template=self.theme.plotly_template)
        fig.update_layout(
            font=dict(family=self.theme.font_family, size=int(self.theme.label_size)),
            title_font_size=int(self.theme.title_size),
            paper_bgcolor=self.theme.background_color,
            plot_bgcolor=self.theme.background_color,
        )
        return fig

    def save_plotly(
        self,
        fig: go.Figure,
        name: str,
        output_dir: Path | str | None = None,
        as_html: bool = True,
        as_image: bool = False,
    ) -> list[Path]:
        """Export a Plotly figure to HTML and/or static image."""
        if output_dir is None:
            output_dir = Path(".")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved: list[Path] = []
        if as_html:
            path = output_dir / f"{name}.html"
            pio.write_html(fig, str(path), include_plotlyjs="cdn")
            saved.append(path)
        if as_image:
            path = output_dir / f"{name}.png"
            try:
                pio.write_image(fig, str(path), width=1200, height=700, scale=2)
                saved.append(path)
            except Exception:
                pass  # kaleido may not be available
        return saved

    # ── Colour helpers ─────────────────────────────────────────────

    @property
    def colors(self) -> list[str]:
        """Shortcut to the categorical palette."""
        return self.theme.categorical

    @property
    def severity_colors(self) -> dict[str, str]:
        return self.theme.severity_colors

    @property
    def grade_colors(self) -> dict[int, str]:
        return self.theme.grade_colors

    def color_for_severity(self, severity: str) -> str:
        return self.severity_colors.get(severity, "#999999")

    def color_for_grade(self, grade: int) -> str:
        return self.grade_colors.get(grade, "#999999")
