#!/usr/bin/env python3
"""Discriminant-validity test for profile-expansion diversity metrics.

Validity claim
--------------
A diversity *distance* is only meaningful if it tracks profile-specific
semantic/behavioral content rather than generic chatter. So, holding the
agenda topic fixed:

    distance(two runs of the SAME short profile)
        <  distance(two runs of DIFFERENT short profiles)

If the within-profile and between-profile distance distributions overlap,
the metric is not capturing profile identity and its "diversity" number is
not interpretable. If they separate cleanly, the distance is valid and
every aggregation built on top of it inherits that validity.

What is tested
--------------
* Semantic family -- ``semantic_diversity`` / ``semantic_diversity_min`` /
  ``semantic_diversity_knn`` all share ONE distance: cosine on
  all-MiniLM-L6-v2 embeddings of per-topic patient text (informative-filtered,
  60-word trimmed). Validating that distance validates all three aggregations.
* Behavior diversity -- Jaccard distance on the per-topic set of *mentioned*
  behavioral attributes (the "combo" signal that ``behavior_diversity``
  aggregates), reusing cached LLM extractions (no API calls).

``profile_alignment`` is intentionally excluded: it is an LLM 1-5 alignment
score, not a distance, so the discriminant-validity framing does not apply.

Why control for topic
---------------------
The metrics are topic-aligned. Comparing *same-topic* pairs isolates the
profile signal from the (trivially large) topic signal, which is the strong
and fair claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from angel_common.paths import REPO_ROOT

PACKAGE_ROOT = REPO_ROOT  # repository root; run scripts with `python -m` from here
import sys


from experiments.profile_expansion.metrics.semantic_diversity import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    _build_embeddings,
    _cosine_distance,
    _record_topic_texts,
)
from experiments.profile_expansion.metrics.behavior_diversity import (  # noqa: E402
    TOPIC_ATTRIBUTE_SCHEMA,
    _deserialize_extracted_attributes,
    _jaccard_distance,
)
from experiments.profile_expansion.evaluate_metrics_record_cache import (  # noqa: E402
    load_record_cache,
    record_cache_key,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Per-topic vector / set representations (reuse each metric's own machinery)
# ---------------------------------------------------------------------------
def build_semantic_topic_vectors(
    records: List[Dict[str, Any]],
    *,
    embedding_cache: Dict[str, np.ndarray],
) -> Dict[str, List[Tuple[Any, np.ndarray]]]:
    """topic_key -> [(profile_id, embedding_vector), ...] for INFORMATIVE runs."""
    # 1) Collect informative per-(record, topic) texts using the metric pipeline.
    per_record: List[Tuple[Any, Dict[str, Dict[str, Any]]]] = []
    all_texts: Set[str] = set()
    for rec in records:
        topics = _record_topic_texts(
            rec,
            turns_per_topic=1,
            max_words_per_topic=60,
            min_words_informative=20,
            min_unique_words_informative=8,
        )
        informative = {tk: item for tk, item in topics.items() if item.get("informative")}
        if informative:
            per_record.append((rec.get("profile_id"), informative))
            for item in informative.values():
                all_texts.add(str(item["text"]))

    # 2) Embed once (cached on disk between runs).
    vectors_by_text = _build_embeddings(
        sorted(all_texts),
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        embedding_cache=embedding_cache,
    )

    # 3) Index by topic.
    topic_vectors: Dict[str, List[Tuple[Any, np.ndarray]]] = defaultdict(list)
    for profile_id, informative in per_record:
        for topic_key, item in informative.items():
            vec = vectors_by_text.get(str(item["text"]))
            if vec is not None:
                topic_vectors[topic_key].append((profile_id, vec))
    return topic_vectors


def build_behavior_topic_sets(
    records: List[Dict[str, Any]],
    *,
    record_cache: Dict[str, Any],
) -> Tuple[
    Dict[str, List[Tuple[Any, Set[str]]]],
    Dict[str, List[Tuple[Any, Dict[str, Set[str]]]]],
]:
    """Return both behavior representations behavior_diversity aggregates.

    combo  : topic_key -> [(profile_id, set of mentioned ATTRIBUTE NAMES)]
             -> "which topics/attributes did the patient touch" (driven largely
                by the fixed therapist agenda).
    values : topic_key -> [(profile_id, {attr: set of extracted VALUE strings})]
             -> "what specifically did the patient say" -- the profile-specific
                content. Only non-empty attributes are kept.

    Only runs whose topic actually had patient text contribute (matching the
    metric's ``topics_with_patient_text`` gate).
    """
    behavior_cache = record_cache.get("behavior_extraction", {})
    combo_sets: Dict[str, List[Tuple[Any, Set[str]]]] = defaultdict(list)
    value_sets: Dict[str, List[Tuple[Any, Dict[str, Set[str]]]]] = defaultdict(list)

    for rec in records:
        key = record_cache_key(rec)
        entry = behavior_cache.get(key)
        if not isinstance(entry, dict):
            continue
        extracted = _deserialize_extracted_attributes(entry.get("extracted"))
        topics_with_text = {
            str(t) for t in (entry.get("topics_with_patient_text") or []) if isinstance(t, str)
        }
        profile_id = rec.get("profile_id")
        for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items():
            if topic_key not in topics_with_text:
                continue
            attrs = spec["attributes"]
            topic_values = extracted.get(topic_key, {})
            mentioned = {attr for attr in attrs if topic_values.get(attr)}
            combo_sets[topic_key].append((profile_id, mentioned))
            value_map = {attr: set(topic_values[attr]) for attr in attrs if topic_values.get(attr)}
            value_sets[topic_key].append((profile_id, value_map))
    return combo_sets, value_sets


def value_jaccard_distance(
    left: Dict[str, Set[str]], right: Dict[str, Set[str]]
) -> Optional[float]:
    """Value-level distance between two runs at a topic.

    Mirrors behavior_diversity's value-diversity component: for each attribute
    that BOTH runs mention (non-empty values), compute Jaccard distance over the
    value strings, then average over those shared attributes. Returns None when
    the two runs share no mentioned attribute (undefined -- excluded, exactly as
    the metric excludes such pairs from its per-attribute lists)."""
    dists: List[float] = []
    for attr, lv in left.items():
        rv = right.get(attr)
        if lv and rv:
            dists.append(_jaccard_distance(lv, rv))
    if not dists:
        return None
    return sum(dists) / len(dists)


# ---------------------------------------------------------------------------
# Within / between distance collection
# ---------------------------------------------------------------------------
def collect_distances(
    topic_items: Dict[str, List[Tuple[Any, Any]]],
    dist_fn,
    *,
    rng: random.Random,
    max_within_per_topic: int,
    max_between_per_topic: int,
) -> Dict[str, Dict[str, List[float]]]:
    """For each topic, return {'within': [...], 'between': [...]} distances.

    within  = pairs of runs from the SAME profile, same topic.
    between = pairs of runs from DIFFERENT profiles, same topic (sampled).
    """
    out: Dict[str, Dict[str, List[float]]] = {}
    for topic_key, items in topic_items.items():
        by_profile: Dict[Any, List[Any]] = defaultdict(list)
        for profile_id, payload in items:
            by_profile[profile_id].append(payload)

        # ---- within-profile pairs ----
        within: List[float] = []
        for payloads in by_profile.values():
            if len(payloads) < 2:
                continue
            pairs = list(combinations(range(len(payloads)), 2))
            if len(pairs) > max_within_per_topic:
                pairs = rng.sample(pairs, max_within_per_topic)
            for i, j in pairs:
                d = dist_fn(payloads[i], payloads[j])
                if d is not None:
                    within.append(d)

        # ---- between-profile pairs (sampled) ----
        between: List[float] = []
        profiles = [p for p, v in by_profile.items() if v]
        if len(profiles) >= 2:
            attempts = 0
            max_attempts = max_between_per_topic * 40
            while len(between) < max_between_per_topic and attempts < max_attempts:
                attempts += 1
                pa, pb = rng.sample(profiles, 2)
                ra = rng.choice(by_profile[pa])
                rb = rng.choice(by_profile[pb])
                d = dist_fn(ra, rb)
                if d is not None:
                    between.append(d)

        if within and between:
            out[topic_key] = {"within": within, "between": between}
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def summarize(within: Sequence[float], between: Sequence[float]) -> Dict[str, float]:
    from scipy.stats import mannwhitneyu

    w = np.asarray(within, dtype=np.float64)
    b = np.asarray(between, dtype=np.float64)
    mean_w, mean_b = float(w.mean()), float(b.mean())

    # Pooled SD for Cohen's d.
    nw, nb = len(w), len(b)
    pooled_var = ((nw - 1) * w.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(1, nw + nb - 2)
    pooled_sd = float(np.sqrt(pooled_var)) if pooled_var > 0 else 0.0
    cohens_d = (mean_b - mean_w) / pooled_sd if pooled_sd > 0 else 0.0

    # AUC: can the distance tell a different-profile pair from a same-profile
    # pair? label 1 = between (different). Equivalent to Mann-Whitney U / (nw*nb).
    try:
        u_stat, p_value = mannwhitneyu(b, w, alternative="greater")
        auc = float(u_stat) / (nw * nb)
    except ValueError:
        p_value = float("nan")
        auc = float("nan")

    return {
        "n_within": nw,
        "n_between": nb,
        "mean_within": mean_w,
        "mean_between": mean_b,
        "median_within": float(np.median(w)),
        "median_between": float(np.median(b)),
        "separation_ratio": (mean_b / mean_w) if mean_w > 0 else float("inf"),
        "cohens_d": cohens_d,
        "auc": auc,
        "mannwhitney_p": float(p_value),
    }


def verdict(stats: Dict[str, float]) -> str:
    auc = stats.get("auc", 0.0)
    p = stats.get("mannwhitney_p", 1.0)
    if p < 0.001 and auc >= 0.70:
        return "STRONG"
    if p < 0.05 and auc >= 0.60:
        return "PASS"
    if p < 0.05 and auc > 0.50:
        return "WEAK"
    return "FAIL"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
PALETTE = {"within": "#2c7fb8", "between": "#d95f0e"}


def fig_histograms(results: Dict[str, Dict], outpath: Path) -> None:
    metrics = list(results.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6.0 * len(metrics), 4.6), squeeze=False)
    for ax, name in zip(axes[0], metrics):
        within = results[name]["pooled"]["within"]
        between = results[name]["pooled"]["between"]
        stats = results[name]["stats"]
        bins = np.linspace(
            0.0,
            max(max(within), max(between)) + 1e-6,
            41,
        )
        ax.hist(within, bins=bins, density=True, alpha=0.6,
                color=PALETTE["within"], label="within-profile")
        ax.hist(between, bins=bins, density=True, alpha=0.6,
                color=PALETTE["between"], label="between-profile")
        ax.axvline(stats["mean_within"], color=PALETTE["within"], ls="--", lw=1.5)
        ax.axvline(stats["mean_between"], color=PALETTE["between"], ls="--", lw=1.5)
        ax.set_title(
            f"{name}\nAUC={stats['auc']:.3f}  d={stats['cohens_d']:.2f}  "
            f"ratio={stats['separation_ratio']:.2f}  [{verdict(stats)}]",
            fontsize=10,
        )
        ax.set_xlabel("pairwise distance")
        ax.set_ylabel("density")
        ax.legend(fontsize=9)
    fig.suptitle(
        "Discriminant validity: within-profile vs between-profile distance",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_per_topic_bars(results: Dict[str, Dict], outpath: Path) -> None:
    metrics = list(results.keys())
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4.6 * len(metrics)), squeeze=False)
    for ax, name in zip(axes[:, 0], metrics):
        per_topic = results[name]["per_topic"]
        topics = list(per_topic.keys())
        means_w, means_b, ci_w, ci_b = [], [], [], []
        for tk in topics:
            w = np.asarray(per_topic[tk]["within"])
            b = np.asarray(per_topic[tk]["between"])
            means_w.append(w.mean())
            means_b.append(b.mean())
            ci_w.append(1.96 * w.std(ddof=1) / np.sqrt(len(w)) if len(w) > 1 else 0.0)
            ci_b.append(1.96 * b.std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else 0.0)
        x = np.arange(len(topics))
        ax.bar(x - 0.2, means_w, 0.4, yerr=ci_w, capsize=2,
               color=PALETTE["within"], label="within-profile")
        ax.bar(x + 0.2, means_b, 0.4, yerr=ci_b, capsize=2,
               color=PALETTE["between"], label="between-profile")
        ax.set_xticks(x)
        ax.set_xticklabels(topics, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("mean distance")
        ax.set_title(f"{name}: per-topic within vs between (95% CI)", fontsize=11)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_conversation_heatmap(
    topic_vectors: Dict[str, List[Tuple[Any, np.ndarray]]],
    *,
    topic_key: str,
    outpath: Path,
    rng: random.Random,
    n_profiles: int = 8,
    runs_per_profile: int = 6,
) -> None:
    """Pairwise cosine-distance heatmap of individual conversations, blocked by
    profile. A valid metric shows a dark (low-distance) block diagonal."""
    items = topic_vectors.get(topic_key)
    if not items:
        # fall back to the most populated topic
        topic_key = max(topic_vectors, key=lambda k: len(topic_vectors[k]))
        items = topic_vectors[topic_key]

    by_profile: Dict[Any, List[np.ndarray]] = defaultdict(list)
    for pid, vec in items:
        by_profile[pid].append(vec)
    eligible = [p for p, v in by_profile.items() if len(v) >= runs_per_profile]
    eligible = sorted(eligible)[:n_profiles]
    if len(eligible) < 2:
        print(f"[heatmap] not enough profiles with >= {runs_per_profile} runs at {topic_key}")
        return

    ordered_vecs: List[np.ndarray] = []
    boundaries: List[int] = []
    labels: List[Any] = []
    for pid in eligible:
        vecs = by_profile[pid][:runs_per_profile]
        ordered_vecs.extend(vecs)
        boundaries.append(len(ordered_vecs))
        labels.append(pid)

    n = len(ordered_vecs)
    mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            mat[i, j] = _cosine_distance(ordered_vecs[i], ordered_vecs[j])

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0)
    fig.colorbar(im, ax=ax, label="cosine distance")
    # block boundaries + per-profile tick labels at block centers
    prev = 0
    centers, edges = [], []
    for b, pid in zip(boundaries, labels):
        centers.append((prev + b - 1) / 2.0)
        edges.append(b - 0.5)
        prev = b
    for e in edges[:-1]:
        ax.axhline(e, color="white", lw=1.0)
        ax.axvline(e, color="white", lw=1.0)
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels([f"p{p}" for p in labels], rotation=90, fontsize=8)
    ax.set_yticklabels([f"p{p}" for p in labels], fontsize=8)
    ax.set_title(
        f"Per-conversation cosine distance @ topic '{topic_key}'\n"
        f"(blocks = profiles; valid metric => dark block diagonal)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_summary(results: Dict[str, Dict], outpath: Path) -> None:
    metrics = list(results.keys())
    aucs = [results[m]["stats"]["auc"] for m in metrics]
    ratios = [results[m]["stats"]["separation_ratio"] for m in metrics]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(metrics))
    ax1.bar(x, aucs, color="#3182bd")
    ax1.axhline(0.5, color="gray", ls="--", lw=1, label="chance (0.5)")
    ax1.set_ylim(0.4, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("AUC (same vs different pair)")
    ax1.set_title("Separation AUC")
    ax1.legend(fontsize=9)
    for xi, v in zip(x, aucs):
        ax1.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax2.bar(x, ratios, color="#31a354")
    ax2.axhline(1.0, color="gray", ls="--", lw=1, label="no separation (1.0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("mean_between / mean_within")
    ax2.set_title("Separation ratio")
    ax2.legend(fontsize=9)
    for xi, v in zip(x, ratios):
        ax2.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=layout.RESULTS_DIR / "patient_psi_agenda_runs25.jsonl",
    )
    ap.add_argument(
        "--record-cache",
        type=Path,
        default=None,
        help="behavior-extraction cache (default: <input>.record_cache.json)",
    )
    ap.add_argument("--outdir", type=Path, default=layout.VALIDITY_DIR)
    ap.add_argument("--embed-cache", type=Path, default=layout.VALIDITY_DIR / "embed_cache.npz")
    ap.add_argument("--max-within-per-topic", type=int, default=4000)
    ap.add_argument("--max-between-per-topic", type=int, default=4000)
    ap.add_argument("--heatmap-topic", type=str, default="presenting_problem")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"[load] {args.input}", flush=True)
    records = load_records(args.input)
    print(f"[load] {len(records)} records", flush=True)

    record_cache_path = args.record_cache or Path(str(args.input) + ".record_cache.json")
    record_cache = load_record_cache(record_cache_path)
    print(
        f"[load] record cache behavior_extraction="
        f"{len(record_cache.get('behavior_extraction', {}))}",
        flush=True,
    )

    # ----- embeddings (persist on disk) -----
    embedding_cache: Dict[str, np.ndarray] = {}
    if args.embed_cache.exists():
        npz = np.load(args.embed_cache, allow_pickle=False)
        embedding_cache = {k: npz[k] for k in npz.files}
        print(f"[embed] loaded {len(embedding_cache)} cached vectors", flush=True)

    print("[semantic] building per-topic embeddings ...", flush=True)
    topic_vectors = build_semantic_topic_vectors(records, embedding_cache=embedding_cache)
    np.savez(args.embed_cache, **embedding_cache)
    print(
        f"[semantic] topics={len(topic_vectors)} "
        f"total_vectors={sum(len(v) for v in topic_vectors.values())}",
        flush=True,
    )

    print("[behavior] building per-topic attribute sets ...", flush=True)
    combo_sets, value_sets = build_behavior_topic_sets(records, record_cache=record_cache)
    print(
        f"[behavior] topics={len(combo_sets)} "
        f"total_runs={sum(len(v) for v in combo_sets.values())}",
        flush=True,
    )

    # ----- collect distances -----
    sem_per_topic = collect_distances(
        topic_vectors, _cosine_distance, rng=rng,
        max_within_per_topic=args.max_within_per_topic,
        max_between_per_topic=args.max_between_per_topic,
    )
    beh_combo_per_topic = collect_distances(
        combo_sets, _jaccard_distance, rng=rng,
        max_within_per_topic=args.max_within_per_topic,
        max_between_per_topic=args.max_between_per_topic,
    )
    beh_value_per_topic = collect_distances(
        value_sets, value_jaccard_distance, rng=rng,
        max_within_per_topic=args.max_within_per_topic,
        max_between_per_topic=args.max_between_per_topic,
    )

    def pool(per_topic: Dict[str, Dict[str, List[float]]]) -> Dict[str, List[float]]:
        within: List[float] = []
        between: List[float] = []
        for d in per_topic.values():
            within.extend(d["within"])
            between.extend(d["between"])
        return {"within": within, "between": between}

    results: Dict[str, Dict] = {}
    results["semantic (cosine)"] = {
        "per_topic": sem_per_topic,
        "pooled": pool(sem_per_topic),
    }
    results["behavior combo (jaccard)"] = {
        "per_topic": beh_combo_per_topic,
        "pooled": pool(beh_combo_per_topic),
    }
    results["behavior value (jaccard)"] = {
        "per_topic": beh_value_per_topic,
        "pooled": pool(beh_value_per_topic),
    }
    for name, payload in results.items():
        payload["stats"] = summarize(payload["pooled"]["within"], payload["pooled"]["between"])

    # ----- report -----
    print("\n================ VALIDITY SUMMARY ================")
    for name, payload in results.items():
        s = payload["stats"]
        print(
            f"\n[{name}]  verdict={verdict(s)}\n"
            f"  n_within={s['n_within']}  n_between={s['n_between']}\n"
            f"  mean_within ={s['mean_within']:.4f}   mean_between ={s['mean_between']:.4f}\n"
            f"  separation_ratio={s['separation_ratio']:.3f}   cohens_d={s['cohens_d']:.3f}\n"
            f"  AUC={s['auc']:.4f}   Mann-Whitney p={s['mannwhitney_p']:.2e}"
        )

    # ----- CSV -----
    csv_path = args.outdir / "validity_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        header = ["metric", "verdict"] + list(next(iter(results.values()))["stats"].keys())
        writer.writerow(header)
        for name, payload in results.items():
            s = payload["stats"]
            writer.writerow([name, verdict(s)] + [s[k] for k in header[2:]])
    print(f"\n[write] {csv_path}", flush=True)

    # ----- figures -----
    fig_histograms(results, args.outdir / "fig1_within_vs_between_hist.png")
    fig_per_topic_bars(results, args.outdir / "fig2_per_topic_bars.png")
    fig_conversation_heatmap(
        topic_vectors, topic_key=args.heatmap_topic,
        outpath=args.outdir / "fig3_conversation_distance_heatmap.png", rng=rng,
    )
    fig_summary(results, args.outdir / "fig4_summary_auc_ratio.png")
    print(f"[write] figures -> {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
