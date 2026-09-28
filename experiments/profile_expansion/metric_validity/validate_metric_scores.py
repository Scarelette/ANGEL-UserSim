#!/usr/bin/env python3
"""Score-level validity test for the three semantic-family diversity metrics.

Unlike the pairwise-distance test (validate_diversity_metrics.py) and the
cluster scatter (plot_conversation_clusters.py), this uses each metric's EXACT
diversity score -- the number its scoring function actually outputs -- and asks
whether that score behaves as a valid diversity measure should:

    score(group of K runs from the SAME profile)        [should be LOW]
        <  score(group of K runs from DIFFERENT profiles) [should be HIGH]

A real "profile group" is a set of repeated runs of one short profile, so its
diversity score should be small. If we instead hand the metric a group of K runs
from K *different* profiles, a valid metric must report a clearly higher score.
The separation of the two score distributions is the validity evidence.

The three metrics share one pairwise cosine distance and differ only in
aggregation, so they get separate figures here (not in the scatter, which would
be identical across them):

    semantic    score_semantic_diversity      mean of ALL pairwise distances
    group       score_group_diversity         mean k-nearest distance
    simulation  score_min_distance_diversity  mean nearest-neighbor distance

Reuses the on-disk embedding cache; the model is loaded at most once to fill any
gaps, then every group is scored from cache.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import random
import sys
from collections import defaultdict
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from angel_common.paths import REPO_ROOT

PACKAGE_ROOT = REPO_ROOT  # repository root; run scripts with `python -m` from here

from experiments.profile_expansion.metrics.semantic_diversity import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    _build_embeddings,
    _record_topic_texts,
    score_semantic_diversity,
)
from experiments.profile_expansion.metrics.semantic_diversity_knn import score_group_diversity  # noqa: E402
from experiments.profile_expansion.metrics.semantic_diversity_min import (  # noqa: E402
    score_min_distance_diversity,
)

import json  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


METRICS = [
    ("semantic", "semantic diversity (all-pairs)", score_semantic_diversity),
    ("group", "group diversity (k-nearest)", score_group_diversity),
    ("simulation", "simulation diversity (nearest-neighbor)", score_min_distance_diversity),
]
PALETTE = {"within": "#2c7fb8", "between": "#d95f0e"}


def load_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def prefill_embeddings(records: List[Dict[str, Any]], cache: Dict[str, np.ndarray]) -> None:
    """Encode any not-yet-cached informative texts once, so per-group scoring
    never triggers a mid-loop model load."""
    all_texts = set()
    for rec in records:
        topics = _record_topic_texts(
            rec, turns_per_topic=1, max_words_per_topic=60,
            min_words_informative=20, min_unique_words_informative=8,
        )
        for item in topics.values():
            if item.get("informative"):
                all_texts.add(str(item["text"]))
    _build_embeddings(sorted(all_texts), embedding_model=DEFAULT_EMBEDDING_MODEL,
                      embedding_cache=cache)


def score_group(
    score_fn: Callable, group_records: List[Dict[str, Any]], cache: Dict[str, np.ndarray]
) -> Optional[float]:
    """Run a metric scorer on one synthetic group; return its diversity score."""
    with contextlib.redirect_stdout(io.StringIO()):
        result = score_fn(group_records, embedding_cache=cache)
    score = result.get("score")
    return float(score) if score is not None else None


def summarize(within: Sequence[float], between: Sequence[float]) -> Dict[str, float]:
    from scipy.stats import mannwhitneyu

    w = np.asarray(within, dtype=np.float64)
    b = np.asarray(between, dtype=np.float64)
    nw, nb = len(w), len(b)
    pooled_var = ((nw - 1) * w.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(1, nw + nb - 2)
    pooled_sd = float(np.sqrt(pooled_var)) if pooled_var > 0 else 0.0
    mean_w, mean_b = float(w.mean()), float(b.mean())
    try:
        u_stat, p_value = mannwhitneyu(b, w, alternative="greater")
        auc = float(u_stat) / (nw * nb)
    except ValueError:
        u_stat, p_value, auc = 0.0, float("nan"), float("nan")
    return {
        "n_within": nw, "n_between": nb,
        "mean_within": mean_w, "mean_between": mean_b,
        "separation_ratio": (mean_b / mean_w) if mean_w > 0 else float("inf"),
        "cohens_d": (mean_b - mean_w) / pooled_sd if pooled_sd > 0 else 0.0,
        "auc": auc, "mannwhitney_p": float(p_value),
    }


def verdict(s: Dict[str, float]) -> str:
    auc, p = s.get("auc", 0.0), s.get("mannwhitney_p", 1.0)
    if p < 0.001 and auc >= 0.70:
        return "STRONG"
    if p < 0.05 and auc >= 0.60:
        return "PASS"
    if p < 0.05 and auc > 0.50:
        return "WEAK"
    return "FAIL"


def fig_one_metric(name: str, label: str, within, between, stats, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    hi = max(max(within), max(between)) + 1e-6
    bins = np.linspace(0.0, hi, 36)
    ax.hist(within, bins=bins, density=True, alpha=0.6, color=PALETTE["within"],
            label="same-profile group")
    ax.hist(between, bins=bins, density=True, alpha=0.6, color=PALETTE["between"],
            label="different-profile group")
    ax.axvline(stats["mean_within"], color=PALETTE["within"], ls="--", lw=1.5)
    ax.axvline(stats["mean_between"], color=PALETTE["between"], ls="--", lw=1.5)
    ax.set_title(
        f"{label}\nexact metric score: same-profile vs different-profile groups\n"
        f"AUC={stats['auc']:.3f}  d={stats['cohens_d']:.2f}  "
        f"ratio={stats['separation_ratio']:.2f}  [{verdict(stats)}]",
        fontsize=11,
    )
    ax.set_xlabel(f"{name} diversity score")
    ax.set_ylabel("density")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {outpath}", flush=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=layout.RESULTS_DIR / "patient_psi_agenda_runs25.jsonl")
    ap.add_argument("--outdir", type=Path, default=layout.VALIDITY_DIR)
    ap.add_argument("--embed-cache", type=Path, default=layout.VALIDITY_DIR / "embed_cache.npz")
    ap.add_argument("--group-size", type=int, default=10, help="K runs per group")
    ap.add_argument("--n-samples", type=int, default=300, help="groups per condition")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"[load] {args.input}", flush=True)
    records = load_records(args.input)
    by_profile: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_profile[rec.get("profile_id")].append(rec)
    profiles = [p for p, runs in by_profile.items() if len(runs) >= args.group_size]
    print(f"[load] {len(records)} records; {len(profiles)} profiles with >= "
          f"{args.group_size} runs", flush=True)

    cache: Dict[str, np.ndarray] = {}
    if args.embed_cache.exists():
        npz = np.load(args.embed_cache, allow_pickle=False)
        cache = {k: npz[k] for k in npz.files}
    print(f"[embed] {len(cache)} cached vectors; prefilling any gaps ...", flush=True)
    prefill_embeddings(records, cache)
    np.savez(args.embed_cache, **cache)

    K = args.group_size

    # Pre-sample the group memberships (shared across metrics for comparability).
    within_groups: List[List[Dict[str, Any]]] = []
    between_groups: List[List[Dict[str, Any]]] = []
    for _ in range(args.n_samples):
        pid = rng.choice(profiles)
        within_groups.append(rng.sample(by_profile[pid], K))
    for i in range(args.n_samples):
        chosen = rng.sample(profiles, K)
        grp = []
        for pid in chosen:
            r = dict(rng.choice(by_profile[pid]))
            r["profile_id"] = f"MIX_{i}"  # force scorer to treat as ONE group
            r["model"] = "MIX"
            grp.append(r)
        between_groups.append(grp)

    summary: Dict[str, Dict[str, float]] = {}
    for name, label, score_fn in METRICS:
        print(f"[score] {name} ...", flush=True)
        within = [score_group(score_fn, g, cache) for g in within_groups]
        between = [score_group(score_fn, g, cache) for g in between_groups]
        within = [v for v in within if v is not None]
        between = [v for v in between if v is not None]
        stats = summarize(within, between)
        summary[name] = stats
        print(
            f"   within mean={stats['mean_within']:.4f}  "
            f"between mean={stats['mean_between']:.4f}  "
            f"AUC={stats['auc']:.3f}  ratio={stats['separation_ratio']:.2f}  "
            f"[{verdict(stats)}]", flush=True,
        )
        fig_one_metric(name, label, within, between, stats,
                       args.outdir / f"score_validity_{name}.png")

    csv_path = args.outdir / "score_validity_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        cols = list(next(iter(summary.values())).keys())
        writer.writerow(["metric", "verdict"] + cols)
        for name, s in summary.items():
            writer.writerow([name, verdict(s)] + [s[c] for c in cols])
    print(f"[write] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
