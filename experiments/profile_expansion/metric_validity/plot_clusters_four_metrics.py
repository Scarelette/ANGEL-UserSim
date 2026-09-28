#!/usr/bin/env python3
"""Four-metric conversation cluster scatter (redraw of clusters_*_tsne_n10).

2x2 grid of t-SNE projections, one panel per diversity metric. Each dot is one
conversation, colored by profile; closer dots = more similar conversations.

NOTE: semantic / group / simulation share the same pairwise cosine distance, so
their scatters are identical (aggregation strategy does not change a 2D
projection). Behavior (value Jaccard) is the distinct one.

Profile colors interpolate the palette of
results/fig/plot_topic_model_bars_clean.py
(eeyore/angel/patient_psi/roleplay_doh) into 10 distinguishable shades.
Reuses helpers from plot_conversation_clusters.py and the on-disk caches.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
from angel_common.paths import REPO_ROOT

PACKAGE_ROOT = REPO_ROOT  # repository root

from experiments.profile_expansion.metric_validity import plot_conversation_clusters as pcc  # noqa: E402
from experiments.profile_expansion.evaluate_metrics_record_cache import load_record_cache  # noqa: E402

import json  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# Shared ACL figure style (same module the heatmap figure uses, so the printed
# font sizes of fig3 and fig4 match exactly).
from experiments.profile_expansion.figures import acl_fig_style as st  # noqa: E402

# Palette from plot_topic_model_bars_clean.py, ordered for a smooth ramp.
PALETTE = ["#1f77b4", "#98df8a", "#ffbe78", "#f28e8c"]

# Printed width the right-hand legend column takes out of the figure.
LEGEND_W_IN = 1.02

# The shared compact ramp (7 / 8 / 7 effective pt) is sized for a dense
# one-column figure; a scatter carries far fewer labels, so it can afford them
# 40% larger without crowding.
FONT_BOOST = 1.4


def load_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def subsample_to(profiles, profile_ids, matrix, runs_per_profile, rng):
    by_profile: Dict[Any, List[int]] = defaultdict(list)
    for idx, pid in enumerate(profile_ids):
        by_profile[pid].append(idx)
    keep_ids, keep_rows = [], []
    for pid in profiles:
        idxs = by_profile.get(pid, [])
        if runs_per_profile > 0 and len(idxs) > runs_per_profile:
            idxs = sorted(rng.sample(idxs, runs_per_profile))
        for i in idxs:
            keep_ids.append(pid)
            keep_rows.append(matrix[i])
    return keep_ids, np.asarray(keep_rows, dtype=np.float32)


# Descriptors matching fig2_per_topic_bars_four_metrics.
TITLES = {
    "semantic": "Semantic",
    "group": "Group",
    "simulation": "Simulation",
    "behavior": "Behavior",
}


def scatter(ax, ids, coords, color_of, label_of, axis_prefix):
    arr = np.asarray(ids, dtype=object)
    for pid in sorted(set(ids), key=str):
        pts = coords[arr == pid]
        ax.scatter(pts[:, 0], pts[:, 1], s=(2.4 * st.SCALE) ** 2, color=color_of[pid],
                   edgecolor="white", linewidth=0.3 * st.SCALE, alpha=0.92,
                   label=label_of[pid], zorder=3)
    ax.set_xlabel(f"{axis_prefix} dim 1")
    ax.set_ylabel(f"{axis_prefix} dim 2")
    # Few ticks: the coordinates are unitless, so the axes only need a scale cue.
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(length=0.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", 3.0 * st.SCALE))
    ax.grid(color=st.GRID_GRAY, linestyle=(0, (1.5, 3)), linewidth=0.6 * st.SCALE,
            zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=layout.RESULTS_DIR / "patient_psi_agenda_runs25.jsonl")
    ap.add_argument("--record-cache", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=layout.VALIDITY_DIR)
    ap.add_argument("--embed-cache", type=Path, default=layout.VALIDITY_DIR / "embed_cache.npz")
    ap.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    ap.add_argument("--n-profiles", type=int, default=10)
    ap.add_argument("--runs-per-profile", type=int, default=20)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--width", choices=["column", "text"], default="text",
                    help="Printed width: one ACL column, or the two-column span.")
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["semantic", "group", "simulation", "behavior"],
        choices=["semantic", "group", "simulation", "behavior"],
        help="Which metric panels to render (in this order).",
    )
    ap.add_argument("--dump", type=Path, default=None,
                    help="Also write the projected coordinates (simulation + behavior) and a "
                    "cosine-distance matrix over the same profiles to this JSON (read by "
                    "the metric-appendix figure).")
    ap.add_argument("--heatmap-runs", type=int, default=6,
                    help="Runs per profile in the dumped distance matrix.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    records = load_records(args.input)
    cache: Dict[str, np.ndarray] = {}
    if args.embed_cache.exists():
        npz = np.load(args.embed_cache, allow_pickle=False)
        cache = {k: npz[k] for k in npz.files}

    # Semantic conversation vectors (shared by the three cosine metrics).
    sem_ids, sem_mat = pcc.build_semantic_vectors(records, embedding_cache=cache)
    np.savez(args.embed_cache, **cache)

    # Behavior value features.
    rc_path = args.record_cache or Path(str(args.input) + ".record_cache.json")
    record_cache = load_record_cache(rc_path)
    beh_ids, beh_mat = pcc.build_behavior_features(records, record_cache=record_cache, level="value")

    # Shared profile set: top-N most-populated profiles in the semantic space.
    counts: Dict[Any, int] = defaultdict(int)
    for pid in sem_ids:
        counts[pid] += 1
    chosen = sorted(counts.keys(), key=lambda p: (-counts[p], str(p)))[:args.n_profiles]

    # Sequential labels + palette-interpolated colors per chosen profile.
    cmap = LinearSegmentedColormap.from_list("ref_palette", PALETTE, N=256)
    color_of = {pid: cmap(i / max(1, len(chosen) - 1)) for i, pid in enumerate(chosen)}
    label_of = {pid: f"profile {i + 1}" for i, pid in enumerate(chosen)}

    sem_sub_ids, sem_sub = subsample_to(chosen, sem_ids, sem_mat, args.runs_per_profile, rng)
    beh_sub_ids, beh_sub = subsample_to(chosen, beh_ids, beh_mat, args.runs_per_profile, rng)

    axis_prefix = "t-SNE" if args.method == "tsne" else "PCA"
    print(f"[project] semantic ({args.method}) ...", flush=True)
    sem_coords = pcc.project(sem_sub, method=args.method, metric="cosine", seed=args.seed)
    print(f"[project] behavior ({args.method}) ...", flush=True)
    beh_coords = pcc.project(beh_sub, method=args.method, metric="jaccard", seed=args.seed)

    if args.dump is not None:
        # Distance matrix over the same profiles, in the same P1..Pn order, taking
        # the first --heatmap-runs subsampled runs of each profile.
        order = {pid: i for i, pid in enumerate(chosen)}
        rows = sorted(range(len(sem_sub_ids)), key=lambda i: order[sem_sub_ids[i]])
        per = defaultdict(list)
        for i in rows:
            if len(per[sem_sub_ids[i]]) < args.heatmap_runs:
                per[sem_sub_ids[i]].append(i)
        keep = [i for pid in chosen for i in per[pid]]
        vecs = sem_sub[keep] / np.linalg.norm(sem_sub[keep], axis=1, keepdims=True)
        dist = np.clip(1.0 - vecs @ vecs.T, 0.0, None)
        np.fill_diagonal(dist, 0.0)
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        args.dump.write_text(json.dumps({
            "method": args.method,
            "seed": args.seed,
            "profiles": [label_of[pid].replace("profile ", "P") for pid in chosen],
            "simulation": {"profile": [order[p] for p in sem_sub_ids],
                           "xy": np.asarray(sem_coords).round(4).tolist()},
            "behavior": {"profile": [order[p] for p in beh_sub_ids],
                         "xy": np.asarray(beh_coords).round(4).tolist()},
            "heatmap": {"profile": [order[sem_sub_ids[i]] for i in keep],
                        "distance": dist.round(5).tolist()},
        }))
        print(f"[write] {args.dump}", flush=True)

    coords_of = {
        "semantic": (sem_sub_ids, sem_coords),
        "group": (sem_sub_ids, sem_coords),
        "simulation": (sem_sub_ids, sem_coords),
        "behavior": (beh_sub_ids, beh_coords),
    }
    panels = [("abcd"[i], TITLES[name], *coords_of[name])
              for i, name in enumerate(args.metrics)]

    n_panels = len(panels)
    ncols = min(2, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    st.apply_style(
        tick=st.FS_TICK_COMPACT * FONT_BOOST,
        label=st.FS_LABEL_COMPACT * FONT_BOOST,
        legend=st.FS_LEGEND_COMPACT * FONT_BOOST,
        title=st.FS_LABEL_COMPACT * FONT_BOOST,
    )
    width_in = st.width_for(args.width)
    # Panels stay square-ish once the right-hand legend column is subtracted.
    panel_h = (width_in - LEGEND_W_IN) * 0.98 / ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=st.size(width_in, panel_h * nrows + 0.16),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02 * st.SCALE, h_pad=0.02 * st.SCALE)
    flat_axes = np.atleast_1d(axes).ravel()
    for ax, (letter, title, ids, coords) in zip(flat_axes, panels):
        scatter(ax, ids, coords, color_of, label_of, axis_prefix)
        ax.set_title(f"({letter}) {title}", loc="left", fontweight="semibold",
                     fontsize=st.pt(st.FS_LABEL_COMPACT * FONT_BOOST),
                     pad=4.0 * st.SCALE)
    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    # Legend in sequential profile order (1, 2, 3, ...), not sorted-by-id order.
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=2.9 * st.SCALE,
               markerfacecolor=color_of[pid], markeredgecolor="white",
               markeredgewidth=0.3 * st.SCALE, label=label_of[pid])
        for pid in chosen
    ]
    fig.legend(
        handles=legend_handles,
        loc="outside center right",
        ncol=1,
        frameon=False,
        handletextpad=0.3,
        labelspacing=0.55,
        borderpad=0.0,
        borderaxespad=0.2,
    )
    out = args.outdir / f"clusters_four_metrics_{args.method}_n10.png"
    st.save(fig, out.with_suffix(".pdf"), out)
    print(f"[write] {out}", flush=True)


if __name__ == "__main__":
    main()
