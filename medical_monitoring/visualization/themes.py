"""
Visualization theme system with 3 preset themes.

Themes:
    fda_regulatory     -- Blue-grey, minimal ink, FDA ST&F 2.0 / CDER style
    academic_conference -- High contrast, ASCO/ESMO/WCLC poster-ready, 300 dpi
    pharma_dashboard   -- Novartis/Roche-inspired safety dashboard, data-dense
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ThemeConfig:
    """Complete theme configuration for a visualization style."""

    name: str = "fda_regulatory"

    # --- Colour Palettes ---
    categorical: list[str] = field(default_factory=list)
    sequential: list[str] = field(default_factory=list)
    diverging: list[str] = field(default_factory=list)
    severity_colors: dict[str, str] = field(default_factory=dict)
    grade_colors: dict[int, str] = field(default_factory=dict)

    # --- Typography ---
    font_family: str = "Arial"
    title_size: float = 14
    label_size: float = 11
    tick_size: float = 9
    annotation_size: float = 8
    font_weight: str = "normal"

    # --- Grid / Axes ---
    grid_alpha: float = 0.3
    grid_style: str = "--"
    spine_visible: bool = True
    background_color: str = "#FFFFFF"
    grid_color: str = "#CCCCCC"

    # --- Legend ---
    legend_loc: str = "best"
    legend_frameon: bool = True
    legend_fontsize: float = 9

    # --- Export ---
    dpi: int = 150
    figure_width: float = 10.0
    figure_height: float = 6.0
    tight_layout: bool = True

    # --- Plotly-specific ---
    plotly_template: str = "plotly_white"


# ──────────────────────────────────────────────────────────────────────
# FDA Regulatory theme – mimics FDA ST&F 2.0 / CDER review materials
# ──────────────────────────────────────────────────────────────────────
FDA_REGULATORY = ThemeConfig(
    name="fda_regulatory",
    categorical=[
        "#003F72", "#5B9BD5", "#8DB4E2", "#A5C8E1",
        "#C6DBEF", "#6BAED6", "#2171B5", "#08519C",
    ],
    sequential=[
        "#F7FBFF", "#DEEBF7", "#C6DBEF", "#9ECAE1",
        "#6BAED6", "#4292C6", "#2171B5", "#084594",
    ],
    diverging=[
        "#B2182B", "#D6604D", "#F4A582", "#FDDBC7",
        "#D1E5F0", "#92C5DE", "#4393C3", "#2166AC",
    ],
    severity_colors={
        "urgent": "#B2182B",
        "high": "#D6604D",
        "medium": "#5B9BD5",
        "low": "#A5C8E1",
    },
    grade_colors={0: "#F7FBFF", 1: "#C6DBEF", 2: "#6BAED6", 3: "#2171B5", 4: "#08306B", 5: "#67000D"},
    font_family="Arial",
    title_size=12,
    label_size=10,
    tick_size=8,
    annotation_size=7,
    font_weight="normal",
    grid_alpha=0.25,
    grid_style="--",
    spine_visible=True,
    background_color="#FFFFFF",
    grid_color="#D0D0D0",
    legend_loc="best",
    legend_frameon=True,
    legend_fontsize=8,
    dpi=150,
    figure_width=10.0,
    figure_height=6.0,
    tight_layout=True,
    plotly_template="plotly_white",
)


# ──────────────────────────────────────────────────────────────────────
# Academic Conference theme – ASCO / ESMO / WCLC poster style
# ──────────────────────────────────────────────────────────────────────
ACADEMIC_CONFERENCE = ThemeConfig(
    name="academic_conference",
    categorical=[
        "#E64B35", "#4DBBD5", "#00A087", "#3C5488",
        "#F39B7F", "#8491B4", "#91D1C2", "#DC0000",
    ],
    sequential=[
        "#FFF5F0", "#FEE0D2", "#FCBBA1", "#FC9272",
        "#FB6A4A", "#EF3B2C", "#CB181D", "#99000D",
    ],
    diverging=[
        "#D73027", "#F46D43", "#FDAE61", "#FEE08B",
        "#D9EF8B", "#A6D96A", "#66BD63", "#1A9850",
    ],
    severity_colors={
        "urgent": "#DC0000",
        "high": "#E64B35",
        "medium": "#4DBBD5",
        "low": "#00A087",
    },
    grade_colors={0: "#F0F0F0", 1: "#4DBBD5", 2: "#00A087", 3: "#F39B7F", 4: "#E64B35", 5: "#DC0000"},
    font_family="Arial",
    title_size=16,
    label_size=13,
    tick_size=11,
    annotation_size=10,
    font_weight="bold",
    grid_alpha=0.2,
    grid_style="-",
    spine_visible=True,
    background_color="#FFFFFF",
    grid_color="#E0E0E0",
    legend_loc="upper right",
    legend_frameon=False,
    legend_fontsize=11,
    dpi=300,
    figure_width=12.0,
    figure_height=7.5,
    tight_layout=True,
    plotly_template="simple_white",
)


# ──────────────────────────────────────────────────────────────────────
# Pharma Dashboard theme – Novartis / Roche safety dashboard style
# ──────────────────────────────────────────────────────────────────────
PHARMA_DASHBOARD = ThemeConfig(
    name="pharma_dashboard",
    categorical=[
        "#0072B2", "#E69F00", "#56B4E9", "#009E73",
        "#F0E442", "#CC79A7", "#D55E00", "#000000",
    ],
    sequential=[
        "#EDFBFF", "#C5EDFA", "#8DD3FA", "#56B4E9",
        "#3192CF", "#0072B2", "#005587", "#003B5C",
    ],
    diverging=[
        "#D55E00", "#E69F00", "#F0E442", "#FFFFFF",
        "#56B4E9", "#0072B2", "#003B5C",
    ],
    severity_colors={
        "urgent": "#D55E00",
        "high": "#E69F00",
        "medium": "#0072B2",
        "low": "#56B4E9",
    },
    grade_colors={0: "#F5F5F5", 1: "#56B4E9", 2: "#009E73", 3: "#E69F00", 4: "#D55E00", 5: "#CC0000"},
    font_family="Segoe UI",
    title_size=13,
    label_size=11,
    tick_size=9,
    annotation_size=8,
    font_weight="normal",
    grid_alpha=0.15,
    grid_style="-",
    spine_visible=False,
    background_color="#FAFAFA",
    grid_color="#E8E8E8",
    legend_loc="upper right",
    legend_frameon=True,
    legend_fontsize=9,
    dpi=200,
    figure_width=14.0,
    figure_height=8.0,
    tight_layout=True,
    plotly_template="plotly_white",
)


# ──────────────────────────────────────────────────────────────────────
# Theme registry
# ──────────────────────────────────────────────────────────────────────
THEMES: dict[str, ThemeConfig] = {
    "fda_regulatory": FDA_REGULATORY,
    "academic_conference": ACADEMIC_CONFERENCE,
    "pharma_dashboard": PHARMA_DASHBOARD,
}


def get_theme(name: str) -> ThemeConfig:
    """Retrieve a registered theme by name."""
    if name not in THEMES:
        available = ", ".join(THEMES.keys())
        raise ValueError(f"Unknown theme '{name}'. Available: {available}")
    return deepcopy(THEMES[name])


def register_theme(name: str, config: ThemeConfig | dict[str, Any]) -> None:
    """Register a custom theme."""
    if isinstance(config, dict):
        base = deepcopy(FDA_REGULATORY)
        for k, v in config.items():
            if hasattr(base, k):
                setattr(base, k, v)
        base.name = name
        THEMES[name] = base
    else:
        config.name = name
        THEMES[name] = deepcopy(config)


def list_themes() -> list[str]:
    """Return names of all registered themes."""
    return list(THEMES.keys())
