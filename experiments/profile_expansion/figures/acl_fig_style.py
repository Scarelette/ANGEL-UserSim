#!/usr/bin/env python3
"""Shared figure style for the ACL-camera-ready profile-expansion figures.

Why the SCALE knob
------------------
An ACL figure is included at a fixed printed width (``\\columnwidth`` = 3.07in,
or ``\\textwidth`` = 6.3in for a ``figure*``). LaTeX scales the PDF to that
width, so what a reader actually sees is ``nominal_fontsize / SCALE``.

Authoring at ``SCALE = 2`` means the standalone PDF has comfortably large
nominal fonts (17-22pt) *and* lands at 8.5-11pt once LaTeX shrinks it to the
column -- i.e. as large as the paper's own body text. Include the figures with

    \\includegraphics[width=\\columnwidth]{...}   % one-column
    \\includegraphics[width=\\textwidth]{...}     % two-column (figure*)

and do not add an extra ``scale=``; the sizes below already account for it.
"""

from __future__ import annotations

from typing import Dict, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# --- ACL geometry (inches, printed size) -------------------------------------
ACL_TEXT_WIDTH = 6.3  # two-column span (figure*)
ACL_COL_WIDTH = 3.07  # single column

SCALE = 2.0  # author at 2x, LaTeX scales back down; see module docstring

# --- Effective (printed) font sizes in pt ------------------------------------
FS_TICK = 8.5
FS_LABEL = 10.0
FS_LEGEND = 9.0
FS_TITLE = 11.0
FS_ANNOT = 8.0

# The compact ramp: 7 / 8 / 7 effective pt, i.e. 14 / 16 / 14 nominal on the
# SCALE=2 canvas. This is the ramp aspect_alignment.pdf uses, and the reference
# every one-column figure in this set is matched against. Because it is defined
# in *printed* pt, a figure authored at ACL_TEXT_WIDTH gets the same nominal and
# the same on-page size as one authored at ACL_COL_WIDTH.
FS_TICK_COMPACT = 7.0
FS_LABEL_COMPACT = 8.0
FS_LEGEND_COMPACT = 7.0


def pt(size: float) -> float:
    """Effective printed pt -> nominal pt on the SCALE-times-larger canvas."""
    return size * SCALE


def size(width_in: float, height_in: float) -> tuple:
    """Printed figure size in inches -> canvas size."""
    return (width_in * SCALE, height_in * SCALE)


# --- Model identity ----------------------------------------------------------
MODEL_ORDER = ["eeyore", "angel", "patient_psi", "roleplay_doh"]

# Original project palette, kept as-is by request. The pale orange and green sit
# below a 3:1 contrast ratio on white, so identity must not rest on color alone:
# every bar is directly labeled on the x axis and Angel additionally carries a
# hatch (see MODEL_HATCH).
MODEL_COLORS: Dict[str, str] = {
    "eeyore": "#ffbe78",
    "angel": "#1f77b4",
    "patient_psi": "#98df8a",
    "roleplay_doh": "#f28e8c",
}

MODEL_DISPLAY: Dict[str, str] = {
    "eeyore": "Eeyore",
    "angel": "Angel",
    "patient_psi": "Patient-psi",
    "roleplay_doh": "Roleplay-doh",
}

# Non-color encoding: the proposed system carries a hatch so it survives
# grayscale printing and CVD readers.
HIGHLIGHT_MODEL = "angel"
MODEL_HATCH: Dict[str, str] = {"angel": "///"}

INK = "#1a1a1a"
INK_MUTED = "#5a5a5a"
AXIS_GRAY = "#8e8e8e"
GRID_GRAY = "#cccccc"


def apply_style(
    tick: float = FS_TICK,
    label: float = FS_LABEL,
    legend: float = FS_LEGEND,
    title: float = FS_TITLE,
) -> None:
    """Install the shared style.

    The defaults suit a two-column (``\\textwidth``) figure. A one-column figure
    is half as wide but carries the same number of labels, so it wants a smaller
    effective size -- pass e.g. ``tick=7, label=8, legend=7``.
    """
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": pt(tick),
            "axes.titlesize": pt(title),
            "axes.labelsize": pt(label),
            "xtick.labelsize": pt(tick),
            "ytick.labelsize": pt(tick),
            "legend.fontsize": pt(legend),
            "axes.edgecolor": AXIS_GRAY,
            "axes.linewidth": 0.7 * SCALE,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": AXIS_GRAY,
            "ytick.color": AXIS_GRAY,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK_MUTED,
            "xtick.major.size": 0.0,
            "ytick.major.size": 2.5 * SCALE,
            "ytick.major.width": 0.7 * SCALE,
            "xtick.major.pad": 1.4 * SCALE,
            "ytick.major.pad": 2.0 * SCALE,
            "hatch.linewidth": 0.6 * SCALE,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,  # TrueType, not Type-3 (camera-ready safe)
            "ps.fonttype": 42,
        }
    )


def apply_compact_style() -> None:
    """The aspect_alignment.pdf ramp: 14 / 16 / 14 nominal pt."""
    apply_style(
        tick=FS_TICK_COMPACT, label=FS_LABEL_COMPACT, legend=FS_LEGEND_COMPACT
    )


def width_for(kind: str) -> float:
    """'column' -> one ACL column, 'text' -> the two-column span."""
    if kind not in ("column", "text"):
        raise ValueError(f"width must be 'column' or 'text', got {kind!r}")
    return ACL_COL_WIDTH if kind == "column" else ACL_TEXT_WIDTH


def tidy_axes(ax) -> None:
    """Recessive grid + spines: only the axes that carry information stay."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(
        axis="y",
        color=GRID_GRAY,
        linestyle=(0, (1.5, 3)),
        linewidth=0.6 * SCALE,
        zorder=0,
    )
    ax.set_axisbelow(True)


def model_legend_handles(models: Iterable[str]) -> list:
    return [
        Patch(
            facecolor=MODEL_COLORS.get(m, "#767676"),
            edgecolor="white",
            linewidth=0.6 * SCALE,
            hatch=MODEL_HATCH.get(m),
            label=MODEL_DISPLAY.get(m, m),
        )
        for m in models
    ]


def thin_ticks(fig, ax, ticks, font_pt: float, axis: str = "y") -> list:
    """Drop ticks that would sit closer together than their own label height.

    A logit axis squeezes its low end (0.6 and 0.7 are ~1/4 the gap between 0.99
    and 0.999), so a fixed candidate list collides as soon as the labels get
    large. Measure the real spacing and keep a legible subset, bottom-up.
    """
    ticks = list(ticks)
    if len(ticks) < 2:
        return ticks
    setter = ax.set_yticks if axis == "y" else ax.set_xticks
    setter(ticks)
    fig.canvas.draw()  # settle the layout so transData is final
    min_gap_px = 1.45 * font_pt * fig.dpi / 72.0
    idx = 1 if axis == "y" else 0
    kept, last = [], None
    for t in ticks:
        point = (0.0, t) if axis == "y" else (t, 0.0)
        pos = float(ax.transData.transform(point)[idx])
        if last is None or abs(pos - last) >= min_gap_px:
            kept.append(t)
            last = pos
    return kept


def draw_axis_break(ax, y_frac: float = 0.0, size_frac: float = 0.016) -> None:
    """Two slashes on the y spine, marking a y-axis that does not start at 0."""
    kwargs = dict(
        transform=ax.transAxes,
        color=AXIS_GRAY,
        clip_on=False,
        linewidth=0.8 * SCALE,
        solid_capstyle="round",
        zorder=10,
    )
    for offset in (0.0, 2.2 * size_frac):
        ax.plot(
            [-size_frac * 0.55, size_frac * 0.55],
            [y_frac + offset - size_frac, y_frac + offset + size_frac],
            **kwargs,
        )


def save(fig, output_pdf, output_png=None) -> None:
    """Write the PDF at exactly the declared canvas size.

    No ``bbox_inches="tight"``: cropping would change the aspect ratio, and with
    it the LaTeX scale factor that the font sizes above are calibrated against.
    Use a constrained layout instead so nothing is clipped.
    """
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf)
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_png, dpi=300)
    plt.close(fig)
