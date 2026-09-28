#!/usr/bin/env python3
"""Stage / aspect results figure, in the same visual system as the overall figure.

  (a) Simulation Diversity by interview stage
  (b) Behavior Diversity by interview stage   (carries the shared stage labels)
  (c) Profile Alignment by profile aspect

Data: ``plot_paper_figures.compute_all_stats`` -- the same numbers as
``plot_stage_aspect_points.py`` and the bar version. Style: ``figures/paper_style.py``,
shared with ``demo-data/final_analysis/plot_combined_summary.py``, so both main
result figures use one palette, one set of model names, one type ramp and one mark.

Drawn at its printed width (7 in): include with width=\\textwidth.

Writes paper/stage_aspect_summary.{pdf,png}.

Usage (from the repository root):
    python -m experiments.profile_expansion.figures.plot_stage_aspect_paper
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from experiments.profile_expansion import layout

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
from experiments.profile_expansion.figures import paper_style as ps  # noqa: E402
from experiments.profile_expansion.figures import plot_paper_figures as pf  # noqa: E402

# pf's stage names with a plain hyphen ("Symp.-Emo."), as the paper text writes them
STAGE_LABELS = {k: v.replace("\u2013", "-") for k, v in pf.TOPIC_LABELS.items()}

# (block, category keys, labels, title, y-limits, y-ticks, tick decimals)
# Upper limits run past 1.00 so markers at exactly 1.0 are not clipped; no tick above 1.
PANELS = [
    ("stage_simulation", pf.TOPIC_ORDER, STAGE_LABELS, "(a) Simulation Diversity",
     (0.10, 0.56), [0.1, 0.2, 0.3, 0.4, 0.5], 1),
    ("stage_behavior", pf.TOPIC_ORDER, STAGE_LABELS, "(b) Behavior Diversity",
     (0.56, 1.035), [0.6, 0.7, 0.8, 0.9, 1.0], 1),
    ("aspect", pf.ASPECT_ORDER, pf.ASPECT_LABELS, "(c) Profile Alignment",
     (0.78, 1.015), [0.80, 0.85, 0.90, 0.95, 1.00], 2),
]

STAGE_SPAN = 0.72   # first to last marker within a stage, in category units
ASPECT_SPAN = 0.36  # five aspects over the same width: keep clusters about as wide
STAGE_MARKER = 5.0  # 14 stages x 4 models: one step under the overall figure's 6.3
STAGE_CAP = 1.8
STAGE_ROTATION = 25  # 14 stages at 7 in: the flattest tilt at which neighbours clear


def draw_panel(ax, stats, keys, labels, title, ylim, yticks, decimals, *, span,
               markersize, cap, show_labels=True, rotation=0):
    n = len(ps.MODELS)
    offsets = (np.arange(n) - (n - 1) / 2) * span / (n - 1)
    for k, key in enumerate(keys):
        for i, m in enumerate(ps.MODELS):
            mean, lo, hi = stats[key][m]
            if not np.isfinite(mean):
                continue
            if not (np.isfinite(lo) and np.isfinite(hi)):
                lo = hi = mean  # no interval (single value): draw the point alone
            ps.mark(ax, k + offsets[i], mean, lo, hi, m, markersize=markersize, cap=cap)

    ax.set_xlim(-0.55, len(keys) - 0.45)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{t:.{decimals}f}" for t in yticks])
    ax.set_xticks(range(len(keys)))
    if show_labels:
        ax.set_xticklabels([labels.get(k, k) for k in keys], rotation=rotation,
                           ha="right" if rotation else "center",
                           rotation_mode="anchor" if rotation else "default",
                           fontsize=ps.TICK_FS, color=ps.TEXT)
    else:
        ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0, pad=2.5)
    ps.panel_title(ax, title)
    ps.clean_axis(ax)
    ax.spines["left"].set_bounds(ylim[0], yticks[-1])


def plot(blocks, out: Path) -> None:
    ps.apply_style()
    fig = plt.figure(figsize=(ps.FIG_WIDTH, 4.75))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 0.72], hspace=0.26,
                          left=0.075, right=0.995, top=0.895, bottom=0.06)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    # the stage-label band under (b) needs its own room before (c)
    pos = axes[2].get_position()
    axes[2].set_position([pos.x0, pos.y0 - 0.08, pos.width, pos.height])

    for ax, (block, keys, labels, title, ylim, yticks, dec) in zip(axes, PANELS):
        stage = keys is pf.TOPIC_ORDER
        draw_panel(ax, blocks[block], keys, labels, title, ylim, yticks, dec,
                   span=STAGE_SPAN if stage else ASPECT_SPAN,
                   markersize=STAGE_MARKER if stage else ps.MARKER_SIZE,
                   cap=STAGE_CAP if stage else ps.CAP,
                   show_labels=block != "stage_simulation",  # (a) shares (b)'s stage axis
                   rotation=STAGE_ROTATION if stage else 0)

    ps.add_legend(fig)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def check_limits(blocks) -> None:
    """Fail loudly if a CI would be clipped by a hand-set y-limit."""
    for block, keys, _, title, (lo_lim, hi_lim), _, _ in PANELS:
        vals = [v for k in keys for m in ps.MODELS for v in blocks[block][k][m]
                if np.isfinite(v)]
        if min(vals) < lo_lim or max(vals) > hi_lim:
            raise ValueError(f"{title}: data [{min(vals):.3f}, {max(vals):.3f}] "
                             f"outside ylim ({lo_lim}, {hi_lim})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clean-dir", type=Path, default=layout.CLEAN_DIR)
    ap.add_argument("--out", type=Path, default=layout.FIG_DIR / "stage_aspect_summary")
    ap.add_argument("--run-number", type=int, default=4)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-adjust-by-informative-run-ratio", action="store_true")
    args = ap.parse_args()

    docs = pf.load_docs(args.clean_dir, args.run_number)
    blocks = pf.compute_all_stats(docs, n_bootstrap=args.n_bootstrap, seed=args.seed,
                                  adjust_by_ratio=not args.no_adjust_by_informative_run_ratio)
    check_limits(blocks)
    plot(blocks, args.out)
    print(f"saved {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
