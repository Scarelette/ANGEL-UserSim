#!/usr/bin/env python3
"""Construct-validity figure: two tests of the two diversity metrics.

  Profile Constraints (diversity should fall as attributes are fixed)
    (a) Simulation Diversity  vs. number of fixed profile attributes
    (b) Behavior Diversity    vs. number of fixed profile attributes
  Between-Profile Discrimination (different profiles > same profile)
    (c) Simulation Diversity  same- vs. different-profile score distributions
    (d) Behavior Diversity    same- vs. different-profile score distributions

The run-count sensitivity analysis is drawn separately (``--runs-out``) for the
appendix: (a) Simulation / (b) Behavior Diversity vs. number of runs.

Data
  (a-b) ``plot_paper_figures.collect_fixattr_stats``. By default this is the
      measured run-4 fixed-attribute metrics.
  (c-d) $ANGEL_OUTPUT_DIR/profile_expansion/metric_validity/final/score_validity_samples.json, written by
      ``plot_validity_compact_combined.py --dump-scores`` (the samples behind
      score_stage_validity.pdf panels a-b).
  runs ``plot_paper_figures.collect_run_stats`` (same numbers as diversity_by_runs.pdf)

Style: ``figures/paper_style.py``, shared with the overall and stage/aspect figures.

Writes <figures>/metric_analysis.{pdf,png} and <figures>/run_count_sensitivity.{pdf,png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from experiments.profile_expansion import layout

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
PKG = HERE.parent  # experiments/profile_expansion
from experiments.profile_expansion.figures import paper_style as ps  # noqa: E402
from experiments.profile_expansion.figures import plot_paper_figures as pf  # noqa: E402

# (label, metric key, per-profile field, skip unscored) -- rows of (a) and (b)
METRICS = [
    ("Simulation Diversity", "min_distance_diversity", "min_distance_diversity", False),
    ("Behavior Diversity", "behavior_diversity", "behavior_diversity", True),
]

# Drawn at its printed width (full-width figure*, \textwidth ~ 7 in), so matplotlib
# points are paper points. Type ramp matches the compact overall-results figure
# (demo-data/final_analysis/plot_combined_summary.py).
FIG_W = 7.0
FIG_H = 2.85
ROW_HEADING_FS = 8.5
PANEL_TITLE_FS = 8.0  # regular weight: only the two row headings are bold
LEGEND_FS = 7.5
LABEL_FS = 7.5   # axis labels, Same / Different profile
TICK_FS = 7.5
ANNOT_FS = 6.8   # AUC / d: a result annotation, kept below the titles
ANNOT_COLOR = "#7a8392"
LEGEND_MARKER = 5.0

LINE_W = 1.3
ANGEL_LINE_W = 1.6  # subtly heavier: the proposed method
OTHER_ALPHA = 0.9
BAND_ALPHA = 0.07
MODEL_BAND_ALPHA = {"eeyore": 0.05}  # its jagged band otherwise draws the eye
LINE_MARKER = 3.8
GRID_ALPHA = 0.18
MARK_EVERY = 2  # 22-25 points per line; every point is visually heavy

# Same / different profile: its own two-colour code, distinct from the model palette.
SAME_COLOR = "#7d9cc0"   # dusty blue
DIFF_COLOR = "#d68fab"   # rose
VALIDITY = [("(c) Simulation Diversity", "simulation"), ("(d) Behavior Diversity", "behavior")]
ROW_HEADINGS = ["Profile Constraints", "Between-Profile Discrimination"]
ROW_HEADING_GAP = 0.02  # inches between the panel titles and the row heading above
PANEL_TITLE_PAD = 3.0
Y_PAD = 0.06          # fraction of the data range added below/above the curves


def draw_curves(ax, stats):
    """One line + CI band per model; stats: model -> x -> (mean, lo, hi)."""
    for m in ps.MODELS:
        by_x = stats.get(m)
        if not by_x:
            continue
        xs = sorted(by_x)
        mean, lo, hi = (np.array([by_x[x][j] for x in xs]) for j in range(3))
        color = ps.MODEL_COLORS[m]
        focus = m == ps.HIGHLIGHT_MODEL
        ax.fill_between(xs, lo, hi, color=color, alpha=MODEL_BAND_ALPHA.get(m, BAND_ALPHA),
                        linewidth=0, zorder=2)
        ax.plot(xs, mean, color=color, alpha=1.0 if focus else OTHER_ALPHA,
                linewidth=ANGEL_LINE_W if focus else LINE_W,
                marker=ps.MODEL_MARKERS[m], markevery=MARK_EVERY, markersize=LINE_MARKER,
                markeredgecolor=ps.EDGE, markeredgewidth=0.4, zorder=4 if focus else 3)
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    y_lo = min(v[1] for by_x in stats.values() for v in by_x.values())
    y_hi = max(v[2] for by_x in stats.values() for v in by_x.values())
    ax.set_ylim(y_lo - Y_PAD * (y_hi - y_lo), y_hi + Y_PAD * (y_hi - y_lo))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
    ps.clean_axis(ax)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.tick_params(axis="x", labelsize=TICK_FS, length=2.5, width=0.7, pad=1.5,
                   colors=ps.SPINE, labelcolor=ps.TEXT)
    ax.tick_params(axis="y", labelsize=TICK_FS)


def draw_trend(ax, stats, title, xlabel):
    draw_curves(ax, stats)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS, color=ps.TEXT, labelpad=1)
    ax.set_title(title, loc="left", fontsize=PANEL_TITLE_FS, fontweight="normal",
                 pad=PANEL_TITLE_PAD, color=ps.TEXT)


def draw_validity(ax, block, title):
    groups = [np.asarray(block["same"]), np.asarray(block["different"])]
    parts = ax.violinplot(groups, positions=[0, 1], widths=0.40, showextrema=False)
    for body, color in zip(parts["bodies"], (SAME_COLOR, DIFF_COLOR)):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.45)
        body.set_linewidth(0.6)
    ax.boxplot(groups, positions=[0, 1], widths=0.13, showfliers=False, patch_artist=True,
               medianprops=dict(color=ps.TEXT, linewidth=1.0),
               boxprops=dict(facecolor="white", edgecolor=ps.EDGE, linewidth=0.6),
               whiskerprops=dict(color=ps.EDGE, linewidth=0.6),
               capprops=dict(color=ps.EDGE, linewidth=0.6))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Same profile", "Different profile"], fontsize=LABEL_FS,
                       color=ps.TEXT)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_xlim(-0.55, 1.55)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.set_title(title, loc="left", fontsize=PANEL_TITLE_FS, fontweight="normal",
                 pad=PANEL_TITLE_PAD, color=ps.TEXT)
    ps.clean_axis(ax)
    ax.grid(axis="y", alpha=GRID_ALPHA)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    # headroom so the corner annotation clears the different-profile whisker
    lo, hi = min(g.min() for g in groups), max(g.max() for g in groups)
    ax.set_ylim(lo - 0.04 * (hi - lo), hi + 0.27 * (hi - lo))
    st = block["stats"]
    ax.text(0.985, 0.97, rf"AUC = {st['auc']:.2f}   $d$ = {st['cohens_d']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=ANNOT_FS,
            color=ANNOT_COLOR)


def profile_legend(fig, x, y, loc):
    return fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=LEGEND_MARKER,
                                      markerfacecolor=c, markeredgecolor=c, label=lab)
                               for c, lab in ((SAME_COLOR, "Same profile"),
                                              (DIFF_COLOR, "Different profile"))],
                      loc=loc, bbox_to_anchor=(x, y), ncol=2, frameon=False,
                      fontsize=LEGEND_FS, handlelength=1.0, handleheight=0.8,
                      handletextpad=0.4, columnspacing=1.2, labelcolor=ps.TEXT)


def model_legend(fig, x, y, loc):
    leg = ps.add_legend(fig, y=y, x=x)
    leg.set_loc(loc)
    for text in leg.get_texts():
        text.set_fontsize(LEGEND_FS)
        text.set_fontweight("normal")  # ANGEL is carried by its blue diamond here
    for handle in leg.legend_handles:
        handle.set_markersize(LEGEND_MARKER)
    return leg


def save(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def row_header(fig, axes, heading, legend=None):
    """Bold heading just above ``axes``' panel titles, flush with the first title;
    ``legend(fig, x, y, loc)`` is centred on the same line, right of the heading."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    fig_h = fig.get_size_inches()[1]
    # set_title(loc="left") draws into _left_title; ax.title is the empty centre one
    title_top = max(inv.transform(ax._left_title.get_window_extent())[1, 1] for ax in axes)
    y = title_top + (ROW_HEADING_GAP + 0.5 * ROW_HEADING_FS / 72) / fig_h
    txt = fig.text(left, y, heading, ha="left", va="center", fontsize=ROW_HEADING_FS,
                   fontweight="bold", color=ps.TEXT)
    if legend is not None:
        heading_right = inv.transform(txt.get_window_extent())[1, 0]
        legend(fig, 0.5 * (heading_right + 0.04 + right), y, "center")
    return y


def plot(fixattr, validity, out: Path) -> None:
    ps.apply_style()
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.80, wspace=0.15,
                            left=0.07, right=0.995, top=0.87, bottom=0.075)
    rows = [[fig.add_subplot(grid[i, j]) for j in range(2)] for i in range(2)]
    for ax, (label, key, _, _), tag in zip(rows[0], METRICS, "ab"):
        draw_trend(ax, fixattr[key], f"({tag}) {label}", "Number of fixed attributes")
    for ax, (title, key) in zip(rows[1], VALIDITY):
        draw_validity(ax, validity[key], title)

    for i, (axes, heading) in enumerate(zip(rows, ROW_HEADINGS)):
        row_header(fig, axes, heading, model_legend if i == 0 else profile_legend)
    save(fig, out)


def plot_runs(runs, out: Path) -> None:
    """Appendix: run-count sensitivity of both metrics."""
    ps.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.2), gridspec_kw=dict(wspace=0.15))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.78, bottom=0.2)
    for ax, (label, key, _, _), tag in zip(axes, METRICS, "ab"):
        draw_trend(ax, runs[key], f"({tag}) {label}", "Number of runs")
    model_legend(fig, 0.5, 0.995, "upper center")
    save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixattr-input-glob", default=str(
        layout.FIXATTR_DIR / "fixattr_variant_*.metrics.run4.json"))
    ap.add_argument("--runs-input-glob", default=str(
        layout.RESULTS_DIR / "*_agenda_runs25.metrics.run*.json"))
    ap.add_argument("--validity-json", type=Path, default=(
        layout.VALIDITY_FINAL_DIR / "score_validity_samples.json"))
    ap.add_argument("--out", type=Path, default=layout.FIG_DIR / "metric_analysis")
    ap.add_argument("--runs-out", type=Path, default=layout.FIG_DIR / "run_count_sensitivity")
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # bootstrap settings match the source figures, so every band is bit-identical
    fixattr = {key: pf.collect_fixattr_stats(args.fixattr_input_glob, key, field, skip,
                                             pf.FIXATTR_N_BOOTSTRAP, args.seed)
               for _, key, field, skip in METRICS}
    runs = {key: pf.collect_run_stats(args.runs_input_glob, key, field, args.n_bootstrap,
                                      args.seed)
            for _, key, field, _ in METRICS}
    validity = json.loads(args.validity_json.read_text())
    plot(fixattr, validity, args.out)
    print(f"saved {args.out.with_suffix('.pdf')}")
    plot_runs(runs, args.runs_out)
    print(f"saved {args.runs_out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
