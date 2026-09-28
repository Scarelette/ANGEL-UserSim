"""Shared visual system for the paper's main result figures.

One place for the palette, model names, markers, type ramp (in printed points,
for figures authored at their 7 in printed width), and the dot + CI mark, so
every figure that imports it reads as the same paper. Used by
``demo-data/final_analysis/plot_combined_summary.py`` (overall results) and
``plot_stage_aspect_points.py`` (stage / aspect results).
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.lines import Line2D

FIG_WIDTH = 7.0  # printed width; include with width=\textwidth

MODELS = ["eeyore", "angel", "patient_psi", "roleplay_doh"]
HIGHLIGHT_MODEL = "angel"
DISPLAY_NAMES = {"eeyore": "Eeyore", "angel": "ANGEL", "patient_psi": "Patient-Psi",
                 "roleplay_doh": "Roleplay-DOH"}
# Softer palette matching the method figures (Figs. 1-3).
MODEL_COLORS = {
    "eeyore": "#f2a65a",
    "angel": "#3b82b6",
    "patient_psi": "#7bc96f",
    "roleplay_doh": "#e98282",
}
MODEL_MARKERS = {"eeyore": "o", "angel": "D", "patient_psi": "s", "roleplay_doh": "^"}

TEXT = "#2f3b4f"        # navy-gray ink of the method figures
TEXT_VALUE = "#6b7482"
TEXT_GROUP = "#697386"
SPINE = "#c8ced7"
GRID = "#cdd2d8"

# Type ramp, printed points.
TICK_FS = 7.5
LABEL_FS = 8.0
TITLE_FS = 8.5
LEGEND_FS = 8.5
VALUE_FS = 5.5
GROUP_FS = 6.8

# Marks. Hierarchy: marker > CI > value label > grid.
MARKER_SIZE = 6.3
CI_LW = 1.0
CAP = 2.3
EDGE = "#3b4658"
EDGE_LW = 0.6


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": TICK_FS,
        "axes.linewidth": 0.7,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def mark(ax, x, mean, lo, hi, model, markersize=MARKER_SIZE, ci_lw=CI_LW, cap=CAP):
    """One model's dot + 95% CI."""
    color = MODEL_COLORS[model]
    ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor=color,
                elinewidth=ci_lw, capsize=cap, capthick=ci_lw, zorder=3)
    ax.plot(x, mean, marker=MODEL_MARKERS[model], markersize=markersize, color=color,
            markeredgecolor=EDGE, markeredgewidth=EDGE_LW, linestyle="none", zorder=5)


def clean_axis(ax, bottom=True):
    """Light dotted y grid, left spine only (plus an optional baseline)."""
    ax.grid(axis="y", color=GRID, alpha=0.6, linestyle=(0, (1.5, 2.5)), linewidth=0.6,
            zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(bottom)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(axis="y", labelsize=TICK_FS, length=2.5, width=0.7, pad=2,
                   colors=SPINE, labelcolor=TEXT)


def panel_title(ax, title):
    ax.set_title(title, loc="left", fontsize=TITLE_FS, fontweight="bold", pad=5, color=TEXT)


def add_legend(fig, y=1.0, x=0.5):
    """The one shared legend: model order fixed, ANGEL bold."""
    handles = [Line2D([], [], marker=MODEL_MARKERS[m], markersize=MARKER_SIZE + 0.5,
                      color=MODEL_COLORS[m], markeredgecolor=EDGE, markeredgewidth=EDGE_LW,
                      linestyle="none", label=DISPLAY_NAMES[m]) for m in MODELS]
    leg = fig.legend(handles=handles, loc="upper center", ncol=len(MODELS), frameon=False,
                     bbox_to_anchor=(x, y), fontsize=LEGEND_FS, handletextpad=0.45,
                     columnspacing=1.4, labelcolor=TEXT)
    for text, m in zip(leg.get_texts(), MODELS):
        if m == HIGHLIGHT_MODEL:
            text.set_fontweight("bold")
    return leg
