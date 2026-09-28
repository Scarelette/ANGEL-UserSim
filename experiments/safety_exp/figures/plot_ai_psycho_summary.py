#!/usr/bin/env python3
"""Two-panel Risk / Safety summary for the prefilled long-context (auto) attack.

Replaces the 14x8in ``ai_psycho_summary.pdf`` (a renamed ``draw_bar_plot`` output
from ../plot_auto_attack_risk_safety_ci.py) with a publication-sized figure:

    (a) Risk Score        (b) Safety Score

Risk and Safety are split into their own panels because they point in opposite
directions -- lower Risk is better, higher Safety is better -- so putting them
side by side in one group invites reading the taller bar as the better one. One
colour per panel also removes the need for a legend.

Input: ``auto_attack_risk_safety_ci_summary.csv`` in this directory, written by
../plot_auto_attack_risk_safety_ci.py. Its means and intervals were checked
against a fresh pass over the 251 raw judge CSVs in results/auto_attack/ and
match to 6 decimals.

Error bars are 95% *normal-approximation* intervals (1.96 x SE over responses,
n = 667-671 per model), not bootstrap intervals -- that is what ``ci95()`` in the
upstream script computes. The caption in ai_psycho_summary_caption.tex says so.

Type size
---------
The canvas is 7.2in wide and crops to ~7.0in, to be included at a two-column
``\\textwidth`` in a ``figure*``. ACL's ``\\textwidth`` is 6.3in, so LaTeX shrinks
the PDF by ~0.9 and every nominal size below prints at ~0.9x: ticks land at ~11.5pt,
comfortably above a 10pt caption. FONT_SCALE is the single knob -- see the note above
it, and change it (not the individual FS_* values) if the target width differs. The
model names stay on one line at this size by being tilted (MODEL_TICK_ROTATION).

This deliberately no longer matches the smaller
profile_expansion/results/fig/alignment/aspect_alignment.pdf ramp (7pt printed
ticks), which read too small against the caption.

Usage (from the repository root):
    python -m experiments.safety_exp.figures.plot_ai_psycho_summary
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from angel_common.paths import OUTPUTS_DIR
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# --- Type ramp -----------------------------------------------------------------
# The five FS_* values below are the requested ramp, and they set the *proportions*
# (ticks 8 : label 9 : title 10 : value 8.5). FONT_SCALE sets the absolute size:
# 1.62 puts the ticks at ~11.5pt printed. Flat one-line model names stop fitting
# above ~1.40 (they overlap by 10px at this scale), so the names are tilted instead
# of wrapped -- see MODEL_TICK_ROTATION. Adjust FONT_SCALE, not the individual
# values, to resize the whole figure's text.
FONT_SCALE = 1.62

FS_BASE = 9.0
FS_TICK = 8.0
FS_LABEL = 9.0
FS_TITLE = 10.0
FS_VALUE = 8.5

INK = "#1a1a1a"
AXIS_GRAY = "0.55"
GRID_GRAY = "0.84"
ERROR_GRAY = "0.25"

# --- Metric identity ---------------------------------------------------------
# One colour per panel: the metrics are not comparable categories, so a shared
# categorical palette would imply a relationship that is not there.
RISK_COLOR = "#E6867A"
SAFETY_COLOR = "#78A9CF"
BAR_EDGE = "0.35"

# (panel title, CSV metric name, bar colour)
PANELS = [
    ("Risk Score", "Risk", RISK_COLOR),
    ("Safety Score", "Safety", SAFETY_COLOR),
]

# CSV model key -> display name. The CSV's title-cased keys ("Gpt-4o") are an
# artifact of the upstream script, not how the models are written.
MODEL_ORDER = ["Gemini-3-pro", "Gpt-4o", "Claude-4.5-opus"]
# All three names stay on one line, tilted rather than wrapped, and each is centred
# on its own tick (ha="center") rather than right-anchored to it. Anchoring drags
# every label 0.59in to the left of the bar it names and leaves nothing on the
# right; centring splits that to 0.06in / 0.22in, which is what makes the row look
# balanced. Centring also spaces adjacent labels evenly, so a shallower tilt clears:
# 12 degrees is comfortable here (flat labels overlap by 10.3px at this ramp), and
# shallow matters because the band's depth grows with the angle -- 0.21in flat,
# 0.51in at 12, 0.80in at 24.
MODEL_TICK_ROTATION = 12
MODEL_DISPLAY = {
    "Gemini-3-pro": "Gemini 3 Pro",
    "Gpt-4o": "GPT-4o",
    "Claude-4.5-opus": "Claude 4.5 Opus",
}

# Both codes are scored on the same 0-3 codebook scale (verified over the raw judge
# CSVs: min 0.0, max 3.0 for both), so the panels share the axis outright
# (sharey=True) -- same limits, same ticks, no chance of drift, and bar heights are
# directly comparable across panels.
YLIM = (0.0, 3.12)  # 3.12, not 3.05: headroom for the 2.88 value label
YTICKS = np.arange(0.0, 3.01, 0.5)
XLIM = (-0.55, 2.55)

BAR_WIDTH = 0.62
VALUE_OFFSET = 0.04  # data units above the upper cap
VALUE_FLOOR = 0.08  # keeps Claude's near-zero Risk label off the baseline
VALUE_CEIL = 3.06  # keeps the tallest bar's label off the panel top

# 7.2 x 3.4 rather than the earlier panoramic 7.0 x 2.8: with the type ramp fixed,
# the width-to-height ratio is what decides whether the fonts look deliberate.
FIG_WIDTH = 7.2
FIG_HEIGHT = 3.4


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": FS_BASE * FONT_SCALE,
            "axes.titlesize": FS_TITLE * FONT_SCALE,
            "axes.labelsize": FS_LABEL * FONT_SCALE,
            "xtick.labelsize": FS_TICK * FONT_SCALE,
            "ytick.labelsize": FS_TICK * FONT_SCALE,
            "axes.linewidth": 0.7,
            "axes.edgecolor": AXIS_GRAY,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": AXIS_GRAY,
            "ytick.color": AXIS_GRAY,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.major.size": 0.0,
            "ytick.major.size": 2.5,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,  # TrueType, not Type-3 (camera-ready safe)
            "ps.fonttype": 42,
        }
    )


def read_summary(path: Path) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
    """(model, metric) -> (mean, ci95 half-width, n)."""
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")
    out: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out[(row["model"], row["metric"])] = (
                float(row["mean_from_csv"]),
                float(row["ci95"]),
                int(row["n"]),
            )
    missing = [
        (m, metric)
        for m in MODEL_ORDER
        for _, metric, _ in PANELS
        if (m, metric) not in out
    ]
    if missing:
        raise KeyError(f"Summary CSV is missing rows for: {missing}")
    return out


def plot(summary: Dict[Tuple[str, str], Tuple[float, float, int]], output_base: Path) -> None:
    apply_style()

    fig, axes = plt.subplots(
        1,
        len(PANELS),
        figsize=(FIG_WIDTH, FIG_HEIGHT),
        sharey=True,
        gridspec_kw={"wspace": 0.30},  # authoritative: beats subplots_adjust
    )
    axes = np.atleast_1d(axes)
    x = np.arange(len(MODEL_ORDER), dtype=float)

    for panel_idx, (ax, (title, metric, color)) in enumerate(zip(axes, PANELS)):
        means = np.array([summary[(m, metric)][0] for m in MODEL_ORDER])
        errors = np.array([summary[(m, metric)][1] for m in MODEL_ORDER])

        ax.bar(
            x,
            means,
            width=BAR_WIDTH,
            color=color,
            edgecolor=BAR_EDGE,
            linewidth=0.8,
            yerr=errors,
            error_kw={
                "elinewidth": 0.9,
                "capsize": 3.5,
                "capthick": 0.9,
                "ecolor": ERROR_GRAY,
            },
            zorder=3,
        )
        for i, (mean, err) in enumerate(zip(means, errors)):
            # Claude's Risk bar is ~1% of the axis, so its label carries the value.
            ax.text(
                x[i],
                min(max(mean + err + VALUE_OFFSET, VALUE_FLOOR), VALUE_CEIL),
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=FS_VALUE * FONT_SCALE,
                zorder=4,
            )

        ax.set_title(
            f"({'ab'[panel_idx]}) {title}", loc="left", pad=7.0, fontweight="bold"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODEL_DISPLAY[m] for m in MODEL_ORDER],
            rotation=MODEL_TICK_ROTATION,
            ha="center",  # centred on the tick; see MODEL_TICK_ROTATION
        )
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)
        ax.set_yticks(YTICKS)
        if panel_idx == 0:  # shared axis: the title goes on the left panel only
            ax.set_ylabel("Mean Score", labelpad=2.0)
        else:  # sharey hides these; the values are worth repeating
            ax.tick_params(axis="y", labelleft=True)

        ax.grid(axis="y", linestyle=":", linewidth=0.7, color="0.82", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS_GRAY)
        ax.spines["left"].set_bounds(YTICKS[0], YTICKS[-1])  # stop at the last tick
        ax.tick_params(axis="x", pad=2.0)

    # right=0.975 leaves room for the widest label, which now extends a little past
    # its own panel on the right instead of hanging off the left. wspace=0.30, not
    # 0.18: that overhang otherwise lands on panel (b)'s "0.0" y tick. wspace is set
    # on the GridSpec above, not here -- subplots_adjust cannot override it.
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.20, top=0.88)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    here = OUTPUTS_DIR / "safety_exp" / "fig"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=here / "auto_attack_risk_safety_ci_summary.csv",
    )
    parser.add_argument("--output", type=Path, default=here / "ai_psycho_summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = read_summary(args.summary_csv)
    plot(summary, args.output)
    print(f"saved {args.output.with_suffix('.pdf')}")
    print("\nMean scores with 95% normal-approximation intervals (1.96 x SE):")
    for _, metric, _ in PANELS:
        print(f"  {metric}")
        for model in MODEL_ORDER:
            mean, ci, n = summary[(model, metric)]
            name = MODEL_DISPLAY[model].replace("\n", " ")  # defensive: no wraps now
            print(f"    {name:16s} {mean:.3f} +/- {ci:.3f}  (n={n})")


if __name__ == "__main__":
    main()
