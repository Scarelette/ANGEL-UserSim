#!/usr/bin/env python3
"""Correlation among the four diversity metrics.

Metrics (per model x profile x stage), read from the combined run4 files:
    semantic    semantic_diversity      -> avg_pairwise_cosine_distance
    group       group_diversity         -> avg_k_nearest_cosine_distance
    simulation  min_distance_diversity  -> avg_nearest_neighbor_cosine_distance
    behavior    behavior_diversity      -> behavior_diversity (per topic)

We measure how the metrics relate using Spearman rank correlation (robust to the
different scales/distributions) plus Pearson, at two granularities:
  1. observation level : every (model, profile, stage) cell -- do conversations
     scored high by one metric score high on another?
  2. stage level        : metric means per stage (14 points) -- do the metrics
     rank the agenda stages the same way?

Outputs heatmaps, a pairwise scatter matrix, and a CSV of all pair correlations.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Dict, List, Mapping, Optional, Tuple

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from experiments.profile_expansion.figures import acl_fig_style as st  # noqa: E402

METRICS = [
    ("semantic", "semantic_diversity", "avg_pairwise_cosine_distance"),
    ("group", "group_diversity", "avg_k_nearest_cosine_distance"),
    ("simulation", "min_distance_diversity", "avg_nearest_neighbor_cosine_distance"),
    ("behavior", "behavior_diversity", "behavior_diversity"),
]
METRIC_LABELS = {
    "semantic": "Semantic\n(all-pairs)",
    "group": "Group\n(k-nearest)",
    "simulation": "Simulation\n(nearest-nb)",
    "behavior": "Behavior\n(value)",
}
SHORT_LABEL = {
    "presenting_problem": "Presenting", "symptoms_emotions": "Sym/Emo",
    "symptoms_behaviors": "Sym/Beh", "symptoms_cognitions": "Sym/Cog",
    "onset_timeline": "Onset", "triggers": "Trigger", "impact_functioning": "Impact",
    "current_coping": "Coping", "social_support": "Support", "past_treatment": "PastTx",
    "risk_suicide_self_harm": "Risk/SI", "risk_harm_others": "Risk/Harm",
    "risk_substance_use": "Risk/Subst", "treatment_goal": "Goal",
}


def parse_model(path: Path) -> str:
    m = re.match(r"(.+)_agenda_runs\d+\.metrics\.combined\.run\d+\.json$", path.name)
    return m.group(1) if m else path.stem


def as_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def metric_topic_map(doc: Mapping[str, Any], section: str, field: str) -> Dict[Tuple[Any, str], float]:
    """(profile_id, topic_key) -> value for one metric."""
    out: Dict[Tuple[Any, str], float] = {}
    sec = doc.get(section)
    if not isinstance(sec, dict):
        return out
    for prof in sec.get("profiles", []) or []:
        if not isinstance(prof, dict):
            continue
        if section == "behavior_diversity" and not prof.get("scored", True):
            continue
        pid = prof.get("profile_id")
        for t in prof.get("topics", []) or []:
            if not isinstance(t, dict):
                continue
            tk = t.get("topic_key")
            val = as_float(t.get(field))
            if isinstance(tk, str) and val is not None:
                out[(pid, tk)] = val
    return out


def build_rows(paths: List[Path]) -> List[Dict[str, Any]]:
    """One row per (model, profile, stage) where ALL four metrics are present."""
    rows: List[Dict[str, Any]] = []
    for path in paths:
        model = parse_model(path)
        doc = json.load(path.open())
        maps = {name: metric_topic_map(doc, section, field) for name, section, field in METRICS}
        # keys present in every metric
        common = set.intersection(*[set(m.keys()) for m in maps.values()])
        for (pid, tk) in sorted(common, key=lambda x: (str(x[0]), x[1])):
            row = {"model": model, "profile_id": pid, "topic_key": tk}
            for name in maps:
                row[name] = maps[name][(pid, tk)]
            rows.append(row)
    return rows


def corr_matrix(data: Dict[str, np.ndarray], method: str, names: List[str]) -> np.ndarray:
    n = len(names)
    mat = np.full((n, n), np.nan)
    fn = spearmanr if method == "spearman" else pearsonr
    for i in range(n):
        for j in range(n):
            a, b = data[names[i]], data[names[j]]
            if a.size >= 3:
                r = fn(a, b)[0]
                mat[i, j] = r
    return mat


def heatmap(ax, mat, title, names: List[str]):
    """`title` is accepted but not drawn -- the LaTeX caption carries it."""
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([METRIC_LABELS[n] for n in names])
    ax.set_yticklabels([METRIC_LABELS[n] for n in names])
    ax.tick_params(length=0.0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    for i in range(len(names)):
        for j in range(len(names)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.6 else "#1a1a1a",
                        fontsize=st.pt(st.FS_LABEL_COMPACT), fontweight="bold")
    return im


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-glob", default=str(
        layout.CLEAN_DIR / "*_agenda_runs5.metrics.combined.run4.json"))
    ap.add_argument("--outdir", type=Path, default=layout.CORRELATION_DIR)
    ap.add_argument(
        "--heatmap-metrics",
        nargs="+",
        default=[m[0] for m in METRICS],
        choices=[m[0] for m in METRICS],
        help="Which metrics to show in the correlation heatmap (in this order).",
    )
    ap.add_argument("--width", choices=["column", "text"], default="column",
                    help="Printed width: one ACL column, or the two-column span.")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    paths = sorted(Path(p) for p in glob(args.input_glob))
    if not paths:
        raise FileNotFoundError(args.input_glob)
    print(f"[load] {len(paths)} files: {[parse_model(p) for p in paths]}", flush=True)

    rows = build_rows(paths)
    names = [m[0] for m in METRICS]
    print(f"[rows] {len(rows)} (model x profile x stage) observations", flush=True)

    # ---- observation-level arrays ----
    obs = {n: np.array([r[n] for r in rows], dtype=float) for n in names}

    # ---- stage-level: mean per stage across all model/profile rows ----
    by_stage: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for n in names:
            by_stage[r["topic_key"]][n].append(r[n])
    stages = sorted(by_stage.keys(), key=lambda s: list(SHORT_LABEL).index(s) if s in SHORT_LABEL else 99)
    stage = {n: np.array([np.mean(by_stage[s][n]) for s in stages], dtype=float) for n in names}

    stage_spear = corr_matrix(stage, "spearman", names)
    stage_pear = corr_matrix(stage, "pearson", names)

    # ---- figure: stage-level heatmap (over the selected metrics) ----
    hm_names = list(args.heatmap_metrics)
    stage_spear_hm = corr_matrix(stage, "spearman", hm_names)
    st.apply_compact_style()
    n_hm = len(hm_names)
    width_in = st.width_for(args.width)
    fig, ax = plt.subplots(
        figsize=st.size(width_in, width_in * 0.80),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.01 * st.SCALE, h_pad=0.01 * st.SCALE)
    # No in-figure title: the caption states the method (Spearman), the level
    # (stage means) and n = {len(stages)} stages.
    im = heatmap(ax, stage_spear_hm, None, hm_names)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"Spearman $\rho$")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.2 * st.SCALE, width=0.7 * st.SCALE)
    out = args.outdir / "metric_correlation_heatmaps.png"
    st.save(fig, out.with_suffix(".pdf"), out)
    print(f"[write] {out}", flush=True)

    # ---- pairwise scatter matrix (stage level) ----
    n = len(names)
    fig, axes = plt.subplots(n, n, figsize=(13, 13))
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                ax.hist(stage[names[i]], bins=8, color="#4c78a8", alpha=0.8)
                ax.set_yticks([])
            else:
                ax.scatter(stage[names[j]], stage[names[i]], s=40, alpha=0.8,
                           color="#444444", edgecolors="white", linewidths=0.5)
                rho = spearmanr(stage[names[j]], stage[names[i]])[0]
                ax.text(0.05, 0.92, f"rho={rho:.2f}", transform=ax.transAxes,
                        fontsize=10, fontweight="bold", color="#b30000",
                        va="top", ha="left")
            if i == n - 1:
                ax.set_xlabel(METRIC_LABELS[names[j]].replace("\n", " "), fontsize=9)
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[names[i]].replace("\n", " "), fontsize=9)
            ax.tick_params(labelsize=7)
    fig.suptitle(f"Pairwise relationships among diversity metrics "
                 f"(each dot = one stage mean, n={len(stages)} stages)", fontsize=13, y=1.0)
    fig.tight_layout()
    out2 = args.outdir / "metric_scatter_matrix.png"
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    fig.savefig(out2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out2}", flush=True)

    # ---- CSV of pair correlations (stage level) ----
    csv_path = args.outdir / "metric_correlation_summary.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric_a", "metric_b", "stage_spearman", "stage_pearson", "n_stages"])
        for i in range(n):
            for j in range(i + 1, n):
                w.writerow([names[i], names[j],
                            f"{stage_spear[i, j]:.4f}", f"{stage_pear[i, j]:.4f}", len(stages)])
    print(f"[write] {csv_path}", flush=True)

    # ---- console summary ----
    print("\n=== Spearman (stage level) ===")
    for i in range(n):
        for j in range(i + 1, n):
            print(f"  {names[i]:>10} ~ {names[j]:<10} rho={stage_spear[i, j]:.3f}")


if __name__ == "__main__":
    main()
