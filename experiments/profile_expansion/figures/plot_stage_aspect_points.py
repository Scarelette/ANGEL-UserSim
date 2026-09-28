#!/usr/bin/env python3
"""Point-plot (dot + 95% CI) version of Figure 2, paper/stage_aspect_performance.

    paper/stage_aspect_performance_point.{pdf,png}

The panel order, the stage and aspect sets and their labels, the y ranges and
ticks, the type ramp, the palette, the group breaks and the save/crop are all
`plot_paper_figures`' own -- this module imports them rather than restating
them, and the data is that module's `compute_all_stats`, so the two versions of
Figure 2 cannot disagree about a number. `plot_paper_figures.py` and the bar PDF
it writes are untouched; this writes a `_point` sibling, so the two can be
compared in place and the LaTeX `\\includegraphics` switched when you have
picked one:

    \\includegraphics[width=\\textwidth]{figs/stage_aspect_performance_point.pdf}

What this module does own is the marks and the layout.

Layout: three stacked panels, one shared x axis
-----------------------------------------------
Two things about the bar version's layout cost the page more than they bought.
Its panel (c) is centred in 0.64 of the width, because five bar *groups*
stretched across the full span would be five very wide bars -- but five marks
have no width to stretch, so (c) here spans exactly what (a) and (b) span, with
the same left edge and the same width, and tightens its clusters instead
(`ASPECT_CLUSTER_SPAN`). And every panel carried its own band of stage labels at
26-40 degrees; (a) and (b) have the *same* x axis, so the stage names are
written once, under (b), and (a) keeps only the tick marks. Aligned edges plus
one set of names means a stage in (a) sits directly over the same stage in (b),
and the figure reads as one unit rather than three plots.

The rest follows from those two: the stage labels flatten to 22 degrees and the
aspect labels to nothing at all (see `STAGE_TICK_ROTATION` for what sets each,
which is not what it looks like), the titles shorten to the metric name and
double as the y-axis names, and the canvas height stops being a number to
choose. It is the sum of the figure's vertical bands, three of which are type
and are measured at the ramp in force (`layout`).

Printed at ACL's 6.3in \\textwidth the figure is ~3.9in tall against the bar
version's ~6.5in, on the same data, with the same printed panel heights and the
same printed marks. What pays for the difference is type: 8.0pt ticks against
the bar version's ~11.9pt (see `FIG_WIDTH`).

What is left of the gap under panel (b) is the stage labels themselves: of its
0.54in printed, 0.35in is the label band and 0.20in is white -- less white than
the 0.23in between (a) and (b). Shorter labels are what would buy more, and
they buy little and cost readability (again, `STAGE_TICK_ROTATION`).

Why a point plot reads better here
----------------------------------
Four bars per stage x 14 stages is 56 bars, and on every stage all four land
inside a narrow band, so most of the ink is the shared part below the
interesting range. A marker encodes position rather than length from a
baseline, so panel (a) can drop its zero floor -- the bar version spends its
bottom fifth on a range no model reaches -- and the 95% CI becomes the dominant
mark instead of a whisker perched on a bar.

Mark language, shared with the other point figures in this set
--------------------------------------------------------------
`plot_points_acl.py` (the one-figure point plots) and this module draw the same
mark, each in its own script's units:

  * the marker set already in `plot_paper_figures` for Figure 3's lines --
    Eeyore circle, Angel diamond, Patient-psi square, Roleplay-doh triangle --
    with Angel one step larger, and drawn above its neighbours;
  * the CI in a deepened tint of the series' own colour, not a neutral gray, so
    the interval belongs to its mark (`deepen`);
  * caps sized from their own marker and never wider than it: the mark is what
    should be seen first, and a T wider than its marker makes a field of these
    read as spikes;
  * a hairline white halo where the interval meets the mark, so the interval
    reads as ending at the marker rather than passing under its fill;
  * a dark ring on every marker -- the palette's fills are pale.

Two things a marker cannot carry that the bar version used: Angel's hatch (no
such thing on a marker) and the white bar edge. Angel keeps its emphasis from
the larger diamond and a bold legend label.

Usage (from the repository root):
    python -m experiments.profile_expansion.figures.plot_stage_aspect_points
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Dict, List, NamedTuple, Sequence, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


from experiments.profile_expansion.figures import plot_paper_figures as pf  # noqa: E402


HERE = Path(__file__).resolve().parent
DEFAULT_CLEAN_DIR = layout.CLEAN_DIR
DEFAULT_OUT_DIR = layout.FIG_DIR

# --- Canvas ------------------------------------------------------------------
# Width, and what widening actually does. The printed width is \textwidth
# either way, so authoring wider creates no printed space: it shrinks the type
# and the marks *within* that width. That is the only lever there is on the
# label tilt. The printed pitch between stage ticks is fixed at ~0.42in by
# \textwidth and 14 stages; the tilt floor is asin(label height / pitch); so
# the only way to flatten the labels is to make the labels smaller against that
# fixed pitch, which is what a wider canvas does.
#
#     W       printed tick   stage-label floor   at 17 degrees
#     8.6in       9.9pt          19.2 deg        collides by 1.6pt
#    10.7in       8.0pt          15.2 deg        clears by 1.6pt
#    12.0in       7.1pt          13.5 deg        clears by 3.5pt
#
# 10.7in buys the flattest tilt (17 degrees, from 22) that keeps the ticks at
# 8pt printed -- smaller than the paper's 10pt body and 9pt captions, which is
# normal for figure ticks, and not smaller than that.
TYPE_WIDTH = 8.6  # the width pf's imported type ramp is tuned for
FIG_WIDTH = 10.7
# Everything below that is plot *geometry* rather than type -- panel heights,
# gutters, marks, hairlines, tick lengths -- is scaled with the canvas, so it
# lands on the page at exactly the size it did at 8.6in. Only the type actually
# changes size, which is the whole point of the widening; and one edit to
# FIG_WIDTH re-tunes the figure instead of leaving twelve stale constants.
GEOM = FIG_WIDTH / TYPE_WIDTH

# Height. Not chosen: summed. Every vertical band of the figure is stated below
# as the inches it needs, `layout` measures the three that depend on type (the
# titles, the legend, the two x-label bands) at the ramp actually in force, and
# the canvas is their total. The bar version's fixed `hspace` had the opposite
# arrangement -- a gap in units of "fraction of the mean panel height" holding a
# band whose real size is a fixed number of points -- which is why its own
# comment has to record that the number must be refitted whenever the canvas or
# the type ramp moves. Here the panels move instead, and nothing has to be
# refitted by hand.
#
# The left gutter is measured too (widest y tick label + tick + its pad): the
# panels no longer carry a y *label*. With titles trimmed to the metric name,
# "(a) Simulation Diversity" sits directly over the axis it names, and a
# "Simulation Diversity" y label beside it says the same thing a second time --
# at 1.3in of panel it is also longer than the panel is tall, so the two y
# labels of (a) and (b) ran into each other. The title is the axis name here.
YTICK_GUTTER_IN = 0.09 * GEOM  # tick length + tick pad + a hair
RIGHT_IN = 0.06 * GEOM
PANEL_AB_IN = 1.30 * GEOM  # plot area, panels (a) and (b)
PANEL_C_IN = 0.86 * GEOM  # five aspects across the full width need less height
# White above a panel title. Panel (a)'s x tick marks point down into this
# band, so it has to clear their 2pt as well as look like a break.
TITLE_LEAD_IN = 0.10 * GEOM
TITLE_PAD_PT = 4.0 * GEOM  # title above the axes top, in pt (`ax.set_title(pad=)`)
LEGEND_GAP_IN = 0.05 * GEOM  # legend to panel (a)'s title
# A title that follows a band of x labels needs less white above it than one
# that follows a panel: the 0.10 lead is there to clear panel (a)'s downward
# tick marks, and under a label band there are none.
TITLE_LEAD_AFTER_LABELS_IN = 0.04 * GEOM
# x tick marks: their length and pad sit between the axes and the label band, so
# the layout adds them to the band rather than carrying a fudge factor for them.
X_TICK_LEN_PT = 2.0 * GEOM
X_TICK_PAD_PT = 2.0 * GEOM
BOTTOM_IN = 0.02 * GEOM
# The legend's keys are marks, not text, and it carries its own internal
# padding; its band is sized from the label text with a margin for both.
LEGEND_SLACK = 1.2
# pf's hairlines -- spines, tick marks, grid -- are set in printed pt for a
# 7.2in canvas, so they are scaled here too (see `apply_style`).
HAIRLINE_PT = 0.7 * GEOM
GRID_PT = 0.6 * GEOM
MARKER_EDGE_PT = 0.6 * GEOM

MARKER_RING = "#2e2e2e"

# The CI tint. At this mark size a #8BD17C hairline is not there at all, and
# dropping value alone is not enough either -- these fills are unsaturated, so a
# merely darkened #FDB366 reads as brown rather than as Eeyore. Hold the hue,
# push the saturation up, pull the value down.
CI_SATURATE = 1.9
CI_VALUE = 0.80

# White halo outside the marker ring, in printed pt. 0.35 separates the mark
# from its interval; at 0.8 the interval is severed into two floating stubs.
HALO_PT = 0.35 * GEOM

# Marks, in pt on this canvas (LaTeX scales them by 0.73 with everything else).
# 14 stages x 4 models across the panel puts ~9pt between neighbouring marks
# here, so panels (a)/(b) take the largest mark that still leaves daylight;
# panel (c) has five aspects and can carry more.
STAGE_MARKER_PT = 3.9 * GEOM
STAGE_CI_LINEWIDTH = 1.05 * GEOM
ASPECT_MARKER_PT = 4.4 * GEOM
LEGEND_MARKER_PT = 4.8 * GEOM
ASPECT_CI_LINEWIDTH = 1.2 * GEOM

# Two facts set these angles, and both are worth stating because the obvious
# reasoning about them is wrong.
#
# When do two right-anchored labels tilted by theta collide? Rotate the plane by
# -theta and both become axis-aligned boxes whose top-right corners sit
# p*cos(theta) apart in x and p*sin(theta) apart in y, each w wide and h tall.
# They overlap iff BOTH  w > p*cos(theta)  and  h > p*sin(theta)  -- so either
# condition alone is enough to be safe. Hence:
#
#   * Flattening is limited by the label's HEIGHT, not its length. Over the 14
#     stages' 41pt pitch the floor is asin(h/p) = 19.3 degrees at this ramp, and
#     abbreviating the long labels does not lower it -- the width escape needs
#     w <= p, i.e. about five characters, which "Presenting" will never be. The
#     floor is 19.2 degrees on an 8.6in canvas and 15.2 degrees on this one (see
#     `FIG_WIDTH`: widening is what lowers it); 17 clears by 1.6pt.
#   * Panel (c)'s five aspects have a 116pt pitch and their widest label is 81pt,
#     so the width condition alone clears: they take no tilt at all. A flat label
#     is the cheapest to read and the cheapest vertically -- the band drops from
#     0.55in to 0.25in, which is most of what panel (c) moved up by.
#
# The band a tilted label costs is w*sin(theta) + h*cos(theta), which grows with
# theta, so the flattest angle that clears is always the cheapest one too.
#
# One value to avoid: exactly 45. `check_figure_text_overlap.corners` recovers a
# rotated label's true box by inverting the envelope equations, and that system
# is singular at 45 -- labels there drop out of the overlap test silently.
STAGE_TICK_ROTATION = 17.0
ASPECT_TICK_ROTATION = 0.0

# The stage groups (symptom / history / risk / goal, per
# `pf.TOPIC_GROUP_BREAKS_AFTER`) as alternating bands instead of the bar
# version's three separator hairlines: the same grouping, but as a soft ground
# the four marks of each stage sit on rather than as more lines competing with
# them. Inset so a white gutter runs between the bands. In panel (a), whose
# stage labels are gone, the bands also carry the category divisions on their
# own.
GROUP_BAND_COLOR = "#f6f6f6"
GROUP_BAND_INSET = 0.06

CAP_FRAC = 0.8  # cap half-width as a fraction of the marker's half-width

# First mark to last mark, in category units. Wider than the bar version's 0.78
# total bar width would allow, because a marker is a point: the cluster can
# spread into the gutter the bars needed, and 0.30 units of white still separate
# one stage's four marks from the next one's.
CLUSTER_SPAN = 0.70
# Panel (c) is now as wide as (a) and (b) but holds five categories instead of
# 14, so a category is 1.7in across. The stage panels' span in category units
# would fling one aspect's four marks 1.2in apart and stop reading as a group;
# this keeps the cluster near their absolute width.
ASPECT_CLUSTER_SPAN = 0.34

ANGEL_SCALE = 16.0 / 13.0  # the summary figure's emphasis ratio
Y_PAD_FRAC = 0.08


def deepen(color: str) -> Tuple[float, float, float]:
    """Same hue, more saturation, less value."""
    hue, sat, val = mcolors.rgb_to_hsv(mcolors.to_rgb(color))
    return tuple(
        mcolors.hsv_to_rgb((hue, min(1.0, sat * CI_SATURATE), val * CI_VALUE))
    )


def ci_color(model: str) -> Tuple[float, float, float]:
    return deepen(pf.MODEL_COLORS.get(model, "#767676"))


def marker_size(model: str, base: float) -> float:
    return base * ANGEL_SCALE if model == pf.HIGHLIGHT_MODEL else base


def draw_group_bands(
    ax, group_keys: Sequence[str], after: Sequence[str] = ()
) -> None:
    """Shade alternate runs of `group_keys`, split after each key in `after`.

    With no `after` -- panel (c), whose five aspects form no groups -- every
    category is its own run, which is the same device one step finer.
    """
    edges = [-0.5]
    for idx, key in enumerate(group_keys):
        if (after and key in after) or (not after and idx < len(group_keys) - 1):
            if idx < len(group_keys) - 1:
                edges.append(idx + 0.5)
    edges.append(len(group_keys) - 0.5)
    for i in range(0, len(edges) - 1, 2):
        ax.axvspan(
            edges[i] + GROUP_BAND_INSET,
            edges[i + 1] - GROUP_BAND_INSET,
            color=GROUP_BAND_COLOR,
            linewidth=0,
            zorder=0,
        )


def point_legend_handles(models: Sequence[str], base_marker_pt: float) -> List[Line2D]:
    """Legend keys that look like the marks on the plot, not like bar patches."""
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=pf.MODEL_MARKERS.get(m, "o"),
            markersize=marker_size(m, base_marker_pt),
            markerfacecolor=pf.MODEL_COLORS.get(m, "#767676"),
            markeredgecolor=MARKER_RING,
            markeredgewidth=MARKER_EDGE_PT,
            label=pf.MODEL_DISPLAY.get(m, m),
        )
        for m in models
    ]


def _view(
    stats: pf.Stats,
    group_keys: Sequence[str],
    ylim: Tuple[float, float],
    yticks: Sequence[float],
) -> Tuple[Tuple[float, float], List[float]]:
    """The zoomed view, and which of the panel's own ticks fall inside it.

    A bar is read as a length from its baseline, so the bar version's floor is
    hand-set per panel (0.0 for simulation diversity) and only ever lowered, by
    `pf._fit_floor`, to keep a whisker from being clipped. A marker is read as a
    position, so the view can close in on the data from both sides -- which is
    most of the point of this version: panel (a) spends its bottom fifth on a
    range no model reaches.

    Headroom on both sides is not optional here. Unlike a bar, a marker sitting
    on the extreme is half-clipped if the view stops there, and several
    behaviour and alignment scores are exactly 1.0, while Eeyore's Personality
    interval ends 0.003 above the bar panel's hand-set floor. The view may
    therefore run past 1.0; the ticks never do, since both metrics are bounded
    at 1, and the tick *values* stay the ones the bar panel chose.
    """
    bounds = [
        value
        for key in group_keys
        for model in pf.MODEL_ORDER
        for value in stats[key][model][1:]
        if np.isfinite(value)
    ]
    if not bounds:
        raise RuntimeError("No finite values to plot.")
    low, high = min(bounds), max(bounds)
    pad = max((high - low) * Y_PAD_FRAC, 1e-3)
    view = (max(0.0, low - pad), high + pad)
    return view, [t for t in yticks if view[0] <= t <= min(1.0, view[1])]


def draw_grouped_points(
    ax,
    group_keys: Sequence[str],
    group_labels: Sequence[str],
    stats: pf.Stats,
    title: str,
    ylim: Tuple[float, float],
    yticks: Sequence[float],
    marker_pt: float,
    ci_linewidth: float,
    tick_decimals: int = 2,
    rotation: float = STAGE_TICK_ROTATION,
    cluster_span: float = CLUSTER_SPAN,
    show_xticklabels: bool = True,
) -> None:
    """`pf.draw_grouped_bars`, with a dot + CI in place of each bar."""
    x = np.arange(len(group_keys), dtype=float)
    n_models = len(pf.MODEL_ORDER)
    step = cluster_span / max(n_models - 1, 1)

    for i, model in enumerate(pf.MODEL_ORDER):
        offsets, means, err_lo, err_hi = [], [], [], []
        for cat_idx, key in enumerate(group_keys):
            mean, ci_low, ci_high = stats[key][model]
            if not np.isfinite(mean):
                continue
            offsets.append(x[cat_idx] - cluster_span / 2.0 + i * step)
            means.append(mean)
            finite = np.isfinite(ci_low) and np.isfinite(ci_high)
            err_lo.append(max(0.0, mean - ci_low) if finite else 0.0)
            err_hi.append(max(0.0, ci_high - mean) if finite else 0.0)

        size = marker_size(model, marker_pt)
        z = 3.0 + (2.0 if model == pf.HIGHLIGHT_MODEL else 0.0)
        container = ax.errorbar(
            offsets,
            means,
            yerr=np.array([err_lo, err_hi], dtype=float),
            fmt=pf.MODEL_MARKERS.get(model, "o"),
            markersize=size,
            markerfacecolor=pf.MODEL_COLORS.get(model, "#767676"),
            markeredgecolor=MARKER_RING,
            markeredgewidth=MARKER_EDGE_PT,
            ecolor=ci_color(model),
            elinewidth=ci_linewidth,
            capsize=CAP_FRAC * size / 2.0,
            capthick=ci_linewidth,
            linestyle="none",
            label=pf.MODEL_DISPLAY.get(model, model),
            zorder=z,
        )
        data_line, caps, bars = container.lines
        data_line.set_zorder(z + 0.5)
        data_line.set_path_effects(
            [pe.withStroke(linewidth=0.6 + 2 * HALO_PT, foreground="white")]
        )
        for bar_col in bars:
            bar_col.set_capstyle("round")
        for cap in caps:
            cap.set_solid_capstyle("round")

    ax.set_xticks(x)
    ax.set_xticklabels(
        group_labels,
        rotation=rotation,
        ha="right" if rotation else "center",
        rotation_mode="anchor" if rotation else "default",
    )
    # Panels (a) and (b) share one x axis, so the stage names are written once,
    # under (b). Panel (a) keeps the tick marks: they, and the group bands, are
    # what tie a cluster of four marks in (a) to the name under the same cluster
    # in (b). (The bar version's ramp sets `xtick.major.size` to 0, since a bar
    # sitting on the tick shows where the category is; a point plot has no such
    # thing, and (a) with neither tick nor name would be a floating field.)
    if not show_xticklabels:
        ax.tick_params(axis="x", labelbottom=False)
    ax.set_xlim(-0.6, len(group_keys) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([f"{t:.{tick_decimals}f}" for t in yticks])
    # The title names the metric and sits directly over the axis it names, so
    # there is no y label; see the note on `YTICK_GUTTER_IN`.
    ax.set_title(title, loc="left", pad=TITLE_PAD_PT, fontweight="semibold")

    pf.tidy_panel(ax)
    ax.grid(axis="y", linestyle=":", linewidth=GRID_PT, color=pf.GRID_GRAY, zorder=1)
    ax.set_axisbelow(False)  # honour the zorder set per artist above
    ax.tick_params(
        axis="x",
        length=X_TICK_LEN_PT,
        width=0.7,
        color=pf.AXIS_GRAY,
        pad=X_TICK_PAD_PT,
    )
    # No white bar edges to chew the bottom spine, so the spine itself is the
    # baseline here (the bar version has to draw it over the bars).
    ax.spines["left"].set_bounds(*ax.get_ylim())
    ax.spines["bottom"].set_bounds(*ax.get_xlim())
    if pf.DRAW_AXIS_BREAKS and ylim[0] > 0.0:
        pf.draw_axis_break(ax)


class Layout(NamedTuple):
    """Where everything goes, once the type has been measured."""

    height: float  # canvas height, in inches
    sim: Tuple[float, float, float, float]  # axes rects, in figure fractions
    beh: Tuple[float, float, float, float]
    align: Tuple[float, float, float, float]
    legend_xy: Tuple[float, float]  # anchor for the shared legend's top centre


def text_extent_in(
    labels: Sequence[str], rotation: float, fontsize: float
) -> Tuple[float, float]:
    """Widest and tallest rendered envelope, in inches, over `labels`.

    Measured, not modelled: a label's tilted band is `w*sin(theta) +
    h*cos(theta)`, and `w` depends on the label set, the font and the ramp. A
    throwaway figure is the cheapest way to ask the same text engine that will
    draw them.
    """
    probe = plt.figure(figsize=(1.0, 1.0))
    renderer = probe.canvas.get_renderer()
    width = height = 0.0
    for label in labels:
        text = probe.text(
            0.5,
            0.5,
            label,
            fontsize=fontsize,
            rotation=rotation,
            ha="right",
            va="top",
            rotation_mode="anchor",
        )
        box = text.get_window_extent(renderer)
        width = max(width, box.width / probe.dpi)
        height = max(height, box.height / probe.dpi)
    plt.close(probe)
    return width, height


def layout(
    stage_labels: Sequence[str],
    aspect_labels: Sequence[str],
    ytick_labels: Sequence[str],
) -> Layout:
    """Stack the figure's vertical bands, top to bottom, and total them.

    Three of the bands are type: the panel titles, the legend, and the two bands
    of rotated x labels. They are measured at the ramp in force rather than
    guessed, so the panels keep their heights when the ramp or the labels change
    and the gaps absorb the difference.

    All three panels get the same left edge and the same width -- panel (c)
    included, which in the bar version is centred in 0.64 of the span because
    five bar groups stretched across the full width would be five very wide
    bars. Five *marks* have no width to stretch, so (c) can span what (a) and
    (b) span and stay a point plot; its cluster tightens instead
    (`ASPECT_CLUSTER_SPAN`). Aligned panel edges mean a stage in (a) sits
    directly above the same stage in (b), and the reader gets one x axis to
    learn instead of three.
    """
    title_text_h = text_extent_in(
        ["(a) Simulation Diversity"], 0.0, pf.FS_TITLE * pf.FONT_SCALE
    )[1]
    title_h = TITLE_LEAD_IN + TITLE_PAD_PT / 72.0 + title_text_h
    title_after_labels_h = (
        TITLE_LEAD_AFTER_LABELS_IN + TITLE_PAD_PT / 72.0 + title_text_h
    )
    tick_offset = (X_TICK_LEN_PT + X_TICK_PAD_PT) / 72.0
    legend_h = LEGEND_SLACK * text_extent_in(
        [pf.MODEL_DISPLAY.get(m, m) for m in pf.MODEL_ORDER],
        0.0,
        pf.FS_LEGEND * pf.FONT_SCALE,
    )[1]
    stage_h = tick_offset + text_extent_in(
        stage_labels, STAGE_TICK_ROTATION, pf.FS_XTICK * pf.FONT_SCALE
    )[1]
    aspect_h = tick_offset + text_extent_in(
        aspect_labels, ASPECT_TICK_ROTATION, pf.FS_XTICK * pf.FONT_SCALE
    )[1]
    left_in = (
        YTICK_GUTTER_IN
        + text_extent_in(ytick_labels, 0.0, pf.FS_YTICK * pf.FONT_SCALE)[0]
    )

    height = (
        legend_h
        + LEGEND_GAP_IN
        + title_h
        + PANEL_AB_IN
        + title_h  # (a) has no label band under it: the gap is (b)'s title
        + PANEL_AB_IN
        + stage_h
        + title_after_labels_h
        + PANEL_C_IN
        + aspect_h
        + BOTTOM_IN
    )

    left = left_in / FIG_WIDTH
    width = (FIG_WIDTH - left_in - RIGHT_IN) / FIG_WIDTH

    def rect(top_in: float, panel_in: float) -> Tuple[float, float, float, float]:
        return (left, (height - top_in - panel_in) / height, width, panel_in / height)

    sim_top = legend_h + LEGEND_GAP_IN + title_h
    beh_top = sim_top + PANEL_AB_IN + title_h
    align_top = beh_top + PANEL_AB_IN + stage_h + title_after_labels_h
    return Layout(
        height=height,
        sim=rect(sim_top, PANEL_AB_IN),
        beh=rect(beh_top, PANEL_AB_IN),
        align=rect(align_top, PANEL_C_IN),
        # Centred on the panels, not on the canvas: the legend belongs to the
        # plots, and the canvas centre is off theirs by half the tick gutter.
        legend_xy=(left + width / 2.0, 1.0),
    )


def plot_stage_aspect_points(blocks: Dict[str, pf.Stats], output_base: Path) -> None:
    pf.apply_style()
    # pf's ramp states its hairlines and its y tick geometry in printed pt for a
    # 7.2in canvas. This one is authored wider, so they are scaled to land on
    # the page at the same size -- otherwise the spines and ticks alone would
    # print 25% finer than the figure they belong to.
    mpl.rcParams.update(
        {
            "axes.linewidth": HAIRLINE_PT,
            "xtick.major.width": HAIRLINE_PT,
            "ytick.major.width": HAIRLINE_PT,
            "ytick.major.size": 2.5 * GEOM,
            "ytick.major.pad": 3.5 * GEOM,
        }
    )

    topic_labels = [pf.TOPIC_LABELS.get(k, k) for k in pf.TOPIC_ORDER]
    aspect_labels = [pf.ASPECT_LABELS.get(k, k) for k in pf.ASPECT_ORDER]

    # The views come first: the left gutter is the width of the widest y tick
    # label these produce, and the layout needs it before the axes exist.
    panels = {
        "stage_simulation": (pf.TOPIC_ORDER, pf.STAGE_SIM_YLIM, pf.STAGE_SIM_YTICKS, 1),
        "stage_behavior": (pf.TOPIC_ORDER, pf.STAGE_BEH_YLIM, pf.STAGE_BEH_YTICKS, 2),
        "aspect": (pf.ASPECT_ORDER, pf.ASPECT_YLIM, pf.ASPECT_YTICKS, 2),
    }
    views = {
        block: _view(blocks[block], keys, ylim, yticks) + (decimals,)
        for block, (keys, ylim, yticks, decimals) in panels.items()
    }
    lay = layout(
        topic_labels,
        aspect_labels,
        [f"{t:.{dec}f}" for _, ticks, dec in views.values() for t in ticks],
    )

    fig = plt.figure(figsize=(FIG_WIDTH, lay.height))
    ax_sim = fig.add_axes(lay.sim)
    ax_beh = fig.add_axes(lay.beh)
    ax_align = fig.add_axes(lay.align)

    # Panel titles name the metric only; "by Stage" / "by Aspect" is the
    # caption's job, and dropping it keeps the three titles the same shape.

    # (a) Simulation Diversity by stage -- no bars, so no zero baseline, and no
    # stage names either: they are written once, under (b).
    ylim, yticks, _ = views["stage_simulation"]
    draw_grouped_points(
        ax_sim,
        pf.TOPIC_ORDER,
        topic_labels,
        blocks["stage_simulation"],
        title="(a) Simulation Diversity",
        ylim=ylim,
        yticks=yticks,
        marker_pt=STAGE_MARKER_PT,
        ci_linewidth=STAGE_CI_LINEWIDTH,
        tick_decimals=1,
        show_xticklabels=False,
    )
    draw_group_bands(ax_sim, pf.TOPIC_ORDER, pf.TOPIC_GROUP_BREAKS_AFTER)

    # (b) Behavior Diversity by stage -- carries the shared stage axis.
    ylim, yticks, _ = views["stage_behavior"]
    draw_grouped_points(
        ax_beh,
        pf.TOPIC_ORDER,
        topic_labels,
        blocks["stage_behavior"],
        title="(b) Behavior Diversity",
        ylim=ylim,
        yticks=yticks,
        marker_pt=STAGE_MARKER_PT,
        ci_linewidth=STAGE_CI_LINEWIDTH,
    )
    draw_group_bands(ax_beh, pf.TOPIC_ORDER, pf.TOPIC_GROUP_BREAKS_AFTER)

    # (c) Profile Alignment by aspect -- five categories, full width, tighter
    # clusters, and a shallower tilt on labels that have room.
    ylim, yticks, _ = views["aspect"]
    draw_grouped_points(
        ax_align,
        pf.ASPECT_ORDER,
        aspect_labels,
        blocks["aspect"],
        title="(c) Profile Alignment",
        ylim=ylim,
        yticks=yticks,
        marker_pt=ASPECT_MARKER_PT,
        ci_linewidth=ASPECT_CI_LINEWIDTH,
        rotation=ASPECT_TICK_ROTATION,
        cluster_span=ASPECT_CLUSTER_SPAN,
    )
    draw_group_bands(ax_align, pf.ASPECT_ORDER)

    pf.add_shared_legend(
        fig, point_legend_handles(pf.MODEL_ORDER, LEGEND_MARKER_PT), y=lay.legend_xy[1]
    )
    legend = fig.legends[-1]
    legend.set_bbox_to_anchor(lay.legend_xy, transform=fig.transFigure)
    # The bar version marks the proposed system with a hatch, which a marker
    # cannot carry; the emphasis lands on the legend label instead.
    for text in legend.get_texts():
        if text.get_text() == pf.MODEL_DISPLAY.get(pf.HIGHLIGHT_MODEL):
            text.set_fontweight("bold")

    pf.save(fig, output_base)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, default=DEFAULT_CLEAN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--run-number", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-adjust-by-informative-run-ratio",
        action="store_true",
        help="Use raw per-stage scores instead of scores * informative_run_ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = pf.load_docs(args.clean_dir, args.run_number)
    blocks = pf.compute_all_stats(
        docs,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        adjust_by_ratio=not args.no_adjust_by_informative_run_ratio,
    )
    base = args.out_dir / "stage_aspect_performance_point"
    plot_stage_aspect_points(blocks, base)
    print(f"[figure 2, points] saved {base.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
