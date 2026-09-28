#!/usr/bin/env python3
"""Appendix figures for the between-profile discrimination test.

  A1  Stage-Level Between-Profile Discrimination
      (a) Simulation Diversity  (b) Behavior Diversity -- per-stage mean score for
      same- vs. different-profile groups (dumbbells).
  A2  Conversation-Level Structure by Patient Profile
      (a) Pairwise Distance -- cosine distance between conversations, ordered by
          profile; (b) Simulation Diversity / (c) Behavior Diversity -- t-SNE of the
          representations each metric scores on. P1..P10 are the same profiles in
          all three panels.

Data (under $ANGEL_OUTPUT_DIR/profile_expansion/metric_validity/final/)
  stage_validity_means.json    ``plot_validity_compact_combined.py --dump-stages``
  conversation_structure.json  ``plot_clusters_four_metrics.py --dump``

Style, sizes and header strips are shared with ``plot_metric_analysis.py`` (the
main-paper construct-validity figure). Drawn at printed width: width=\\textwidth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from experiments.profile_expansion import layout

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from experiments.profile_expansion.figures import plot_metric_analysis as ma
from experiments.profile_expansion.figures.plot_metric_analysis import ps

HERE = Path(__file__).resolve().parent
FINAL = layout.VALIDITY_FINAL_DIR

METRICS = [("Simulation Diversity", "simulation"), ("Behavior Diversity", "behavior")]
STAGE_SEPARATORS = [6, 9, 12]  # rows after which the interview moves to a new phase
# Appendix type ramp: panel titles and legends 0.5 pt under the main figure's, so
# the strip heading leads; ticks / stage labels stay at the main figure's 8.5 pt.
TITLE_FS = 9.2
LEGEND_FS = 8.5
BEHAVIOR_XLIM = (0.72, 1.015)  # scores bunch near 1.0; don't let autoscale pad further
DUMBBELL_MARKER = 5.5
CONNECTOR = "#c8ced7"
CONNECTOR_W = 1.2
APP_ROW_HEADING_FS = 10.25
APP_ROW_BAND_H = 0.11  # in; slightly slimmer than main-figure strips

# Muted categorical palette (Tableau 10); deliberately unlike the model colours.
PROFILE_COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
                  "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


def appendix_row_header(fig, axes, heading, legend=None):
    """Slim pale strip + heading, stylistically matched to the main figure."""
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    fig_h = fig.get_size_inches()[1]
    band_h = APP_ROW_BAND_H / fig_h
    title_top = max(inv.transform(ax._left_title.get_window_extent())[1, 1] for ax in axes)
    y = title_top + 0.028 / fig_h + band_h / 2
    fig.patches.append(Rectangle((left - 0.012, y - band_h / 2), right - left + 0.017,
                                 band_h, transform=fig.transFigure, facecolor=ma.ROW_BAND,
                                 edgecolor="none", zorder=0))
    txt = fig.text(left, y, heading, ha="left", va="center", fontsize=APP_ROW_HEADING_FS,
                   fontweight="bold", color=ps.TEXT)
    if legend is not None:
        heading_right = inv.transform(txt.get_window_extent())[1, 0]
        legend(fig, 0.5 * (heading_right + 0.04 + right), y, "center")
    return y


def stage_panel(ax, stages, block, title, show_labels, xlim=None):
    same, diff = np.asarray(block["same"]), np.asarray(block["different"])
    y = np.arange(len(stages))
    ax.hlines(y, np.minimum(same, diff), np.maximum(same, diff), color=CONNECTOR,
              linewidth=CONNECTOR_W, zorder=2)
    for vals, color in ((same, ma.SAME_COLOR), (diff, ma.DIFF_COLOR)):
        ax.plot(vals, y, "o", markersize=DUMBBELL_MARKER, color=color,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    for i in STAGE_SEPARATORS:
        ax.axhline(i - 0.5, color=ps.GRID, linewidth=0.45, alpha=0.3, zorder=1)
    ax.set_ylim(len(stages) - 0.5, -0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(stages if show_labels else [], fontsize=ma.TICK_FS, color=ps.TEXT)
    lo, hi = min(same.min(), diff.min()), max(same.max(), diff.max())
    ax.set_xlim(xlim or (lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax.set_xlabel("Diversity score", fontsize=ma.LABEL_FS, color=ps.TEXT, labelpad=1)
    ax.set_title(title, loc="left", fontsize=TITLE_FS, fontweight="bold",
                 pad=ma.PANEL_TITLE_PAD, color=ps.TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ps.SPINE)
        ax.spines[side].set_linewidth(0.7)
    ax.grid(axis="x", color=ps.GRID, alpha=ma.GRID_ALPHA, linestyle=(0, (1.5, 2.5)),
            linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=ma.TICK_FS, length=2.5, width=0.7, pad=1.5,
                   colors=ps.SPINE, labelcolor=ps.TEXT)
    ax.tick_params(axis="y", length=0, pad=3)


def profile_legend(fig, x, y, loc):
    leg = ma.profile_legend(fig, x, y, loc)
    for text in leg.get_texts():
        text.set_fontsize(LEGEND_FS)
    return leg


def plot_stages(data, out: Path) -> None:
    ps.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(ma.FIG_W, 3.3), gridspec_kw=dict(wspace=0.08))
    fig.subplots_adjust(left=0.105, right=0.99, top=0.855, bottom=0.12)
    for i, (ax, (label, key)) in enumerate(zip(axes, METRICS)):
        stage_panel(ax, data["stages"], data[key], f"({'ab'[i]}) {label}", show_labels=i == 0,
                    xlim=BEHAVIOR_XLIM if key == "behavior" else None)
    appendix_row_header(fig, axes, "Stage-Level Between-Profile Discrimination", profile_legend)
    ma.save(fig, out)


def heatmap_panel(fig, ax, block, labels, title):
    dist = np.asarray(block["distance"])
    prof = np.asarray(block["profile"])
    n = len(dist)
    im = ax.imshow(dist, cmap="viridis", vmin=0.0, interpolation="nearest",
                   extent=(-0.5, n - 0.5, n - 0.5, -0.5))
    edges = np.flatnonzero(np.diff(prof)) + 0.5
    for e in edges:
        ax.axhline(e, color="white", linewidth=0.8, alpha=0.8)
        ax.axvline(e, color="white", linewidth=0.8, alpha=0.8)
    bounds = np.concatenate([[-0.5], edges, [n - 0.5]])
    centers = 0.5 * (bounds[:-1] + bounds[1:])
    ax.set_yticks(centers)
    ax.set_yticklabels([labels[p] for p in np.unique(prof)], fontsize=ma.TICK_FS,
                       color=ps.TEXT)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=TITLE_FS, fontweight="bold",
                 pad=ma.PANEL_TITLE_PAD, color=ps.TEXT)
    return im


def distance_colorbar(fig, cax, im):
    """Slim vertical colour bar right of the heatmap."""
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_visible(False)
    cax.tick_params(labelsize=ma.TICK_FS, length=2, width=0.6, pad=1.5, colors=ps.SPINE,
                    labelcolor=ps.TEXT)
    cax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    cbar.set_label("Cosine distance", rotation=90, fontsize=ma.TICK_FS, color=ps.TEXT,
                   labelpad=5)


def embedding_panel(ax, block, title):
    xy = np.asarray(block["xy"])
    prof = np.asarray(block["profile"])
    for p in np.unique(prof):
        pts = xy[prof == p]
        ax.scatter(pts[:, 0], pts[:, 1], s=18, color=PROFILE_COLORS[p], alpha=0.8,
                   edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_color(ps.SPINE)
        spine.set_linewidth(0.7)
    ax.set_title(title, loc="left", fontsize=TITLE_FS, fontweight="bold",
                 pad=ma.PANEL_TITLE_PAD, color=ps.TEXT)


def plot_structure(data, out: Path) -> None:
    ps.apply_style()
    labels = data["profiles"]
    fig = plt.figure(figsize=(ma.FIG_W, 2.75))
    # columns: heatmap | colour bar | gap for its ticks | two embeddings
    grid = fig.add_gridspec(1, 5, width_ratios=[0.88, 0.03, 0.06, 1.0, 1.0], wspace=0.06,
                            left=0.045, right=0.995, top=0.83, bottom=0.14)
    axes = [fig.add_subplot(grid[0, j]) for j in (0, 3, 4)]
    cax = fig.add_subplot(grid[0, 1])
    im = heatmap_panel(fig, axes[0], data["heatmap"], labels, "(a) Pairwise Distance")
    for ax, (label, key), tag in zip(axes[1:], METRICS, "bc"):
        embedding_panel(ax, data[key], f"({tag}) {label}")
    distance_colorbar(fig, cax, im)
    appendix_row_header(fig, axes, "Conversation-Level Structure by Patient Profile")
    # one profile legend under the two embedding panels (colours are not used in (a))
    x_mid = 0.5 * (axes[1].get_position().x0 + axes[2].get_position().x1)
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=5,
                               markerfacecolor=PROFILE_COLORS[i], markeredgecolor="white",
                               markeredgewidth=0.25, label=lab)
                        for i, lab in enumerate(labels)],
               loc="upper center", bbox_to_anchor=(x_mid, 0.115), ncol=len(labels),
               frameon=False, fontsize=LEGEND_FS, handlelength=0.8,
               handletextpad=0.25, columnspacing=0.8, labelcolor=ps.TEXT)
    ma.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages-json", type=Path, default=FINAL / "stage_validity_means.json")
    ap.add_argument("--structure-json", type=Path,
                    default=FINAL / "conversation_structure.json")
    ap.add_argument("--stages-out", type=Path,
                    default=FINAL / "stage_profile_discrimination")
    ap.add_argument("--structure-out", type=Path,
                    default=FINAL / "conversation_profile_structure")
    args = ap.parse_args()
    plot_stages(json.loads(args.stages_json.read_text()), args.stages_out)
    print(f"saved {args.stages_out.with_suffix('.pdf')}")
    plot_structure(json.loads(args.structure_json.read_text()), args.structure_out)
    print(f"saved {args.structure_out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
