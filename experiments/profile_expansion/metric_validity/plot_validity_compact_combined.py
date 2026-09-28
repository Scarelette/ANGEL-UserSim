#!/usr/bin/env python3
"""Generate two revised validity figures for semantic/behavior diversity.

Figure 1 (2x2):
  (a) Semantic score-level validity
  (b) Behavior score-level validity
  (c) Semantic validity by stage
  (d) Behavior validity by stage

Figure 2 (single panel):
  Conversation-level distance matrix heatmap.

The script reuses cached embeddings and cached behavior extractions. No LLM
calls are made.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from angel_common.paths import REPO_ROOT

PACKAGE_ROOT = REPO_ROOT  # repository root; run scripts with `python -m` from here

from experiments.profile_expansion.evaluate_metrics_record_cache import (  # noqa: E402
    load_record_cache,
    record_cache_key,
)
from experiments.profile_expansion.metrics.behavior_diversity import (  # noqa: E402
    TOPIC_ATTRIBUTE_SCHEMA,
    _deserialize_extracted_attributes,
    _jaccard_distance,
)
from experiments.profile_expansion.metrics.common import mean as _mean  # noqa: E402
from experiments.profile_expansion.metrics.semantic_diversity import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    _record_topic_texts,
    score_semantic_diversity,
)
from experiments.profile_expansion.metrics.semantic_diversity_knn import score_group_diversity  # noqa: E402
from experiments.profile_expansion.metrics.semantic_diversity_min import (  # noqa: E402
    score_min_distance_diversity,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402


COLORS = {
    "same": "#6FA4D9",
    "different": "#EFA1A1",
}

TOPIC_ORDER = [
    "presenting_problem",
    "symptoms_emotions",
    "symptoms_behaviors",
    "symptoms_cognitions",
    "onset_timeline",
    "triggers",
    "impact_functioning",
    "current_coping",
    "social_support",
    "past_treatment",
    "risk_suicide_self_harm",
    "risk_harm_others",
    "risk_substance_use",
    "treatment_goal",
]

TOPIC_DISPLAY = {
    "presenting_problem": "Presenting",
    "symptoms_emotions": "Symp.–Emo.",
    "symptoms_behaviors": "Symp.–Beh.",
    "symptoms_cognitions": "Symp.–Cog.",
    "onset_timeline": "Onset",
    "triggers": "Trigger",
    "impact_functioning": "Impact",
    "current_coping": "Coping",
    "social_support": "Support",
    "past_treatment": "Past Tx",
    "risk_suicide_self_harm": "Suicide",
    "risk_harm_others": "Harm",
    "risk_substance_use": "Substance",
    "treatment_goal": "Goals",
}

STAGE_SEPARATOR_INDICES = [6, 9, 12]

SEMANTIC_SCORE_FUNCS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "semantic": score_semantic_diversity,
    "group": score_group_diversity,
    "simulation": score_min_distance_diversity,
}

SEMANTIC_NAME = {
    "semantic": "all-pairs cosine",
    "group": "k-nearest cosine",
    "simulation": "nearest-neighbor cosine",
}


def load_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def relabel(group: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    out = []
    for rec in group:
        r = dict(rec)
        r["profile_id"] = tag
        r["model"] = "MIX"
        out.append(r)
    return out


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
        p_value, auc = float("nan"), float("nan")
    return {
        "n_within": nw,
        "n_between": nb,
        "mean_within": mean_w,
        "mean_between": mean_b,
        "cohens_d": (mean_b - mean_w) / pooled_sd if pooled_sd > 0 else 0.0,
        "auc": auc,
        "mannwhitney_p": float(p_value),
    }


def semantic_score(
    score_fn: Callable[..., Dict[str, Any]],
    group: List[Dict[str, Any]],
    cache: Dict[str, np.ndarray],
) -> Optional[float]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = score_fn(group, embedding_cache=cache)
    score = result.get("score")
    return float(score) if score is not None else None


def semantic_per_topic(
    score_fn: Callable[..., Dict[str, Any]],
    group: List[Dict[str, Any]],
    cache: Dict[str, np.ndarray],
) -> Dict[str, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = score_fn(group, embedding_cache=cache)
    profs = result.get("profiles") or []
    if not profs:
        return {}
    out: Dict[str, float] = {}
    for topic in profs[0].get("topics", []):
        value = topic.get("avg_pairwise_cosine_distance")
        if value is not None:
            out[str(topic.get("topic_key"))] = float(value)
    return out


def behavior_value_score(
    group: List[Dict[str, Any]],
    key_to_entry: Dict[str, Tuple[Dict[str, Dict[str, Set[str]]], Set[str]]],
) -> Optional[float]:
    runs = []
    for rec in group:
        entry = key_to_entry.get(record_cache_key(rec))
        if entry is not None:
            runs.append(entry)
    if len(runs) < 2:
        return None

    topic_value_divs: List[float] = []
    for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items():
        present = [extracted for (extracted, tset) in runs if topic_key in tset]
        if len(present) < 2:
            continue
        attr_divs: List[float] = []
        for attr in spec["attributes"]:
            value_sets = [
                extracted[topic_key].get(attr, set())
                for extracted in present
                if extracted[topic_key].get(attr)
            ]
            pair = [_jaccard_distance(a, b) for a, b in combinations(value_sets, 2)]
            if pair:
                attr_divs.append(_mean(pair))
        if attr_divs:
            topic_value_divs.append(_mean(attr_divs))
    if not topic_value_divs:
        return None
    return _mean(topic_value_divs)


def behavior_per_topic(
    group: List[Dict[str, Any]],
    key_to_entry: Dict[str, Tuple[Dict[str, Dict[str, Set[str]]], Set[str]]],
) -> Dict[str, float]:
    runs = []
    for rec in group:
        entry = key_to_entry.get(record_cache_key(rec))
        if entry is not None:
            runs.append(entry)
    if len(runs) < 2:
        return {}

    out: Dict[str, float] = {}
    for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items():
        present = [extracted for (extracted, tset) in runs if topic_key in tset]
        if len(present) < 2:
            continue
        attr_divs: List[float] = []
        for attr in spec["attributes"]:
            value_sets = [
                extracted[topic_key].get(attr, set())
                for extracted in present
                if extracted[topic_key].get(attr)
            ]
            pair = [_jaccard_distance(a, b) for a, b in combinations(value_sets, 2)]
            if pair:
                attr_divs.append(_mean(pair))
        if attr_divs:
            out[topic_key] = _mean(attr_divs)
    return out


def semantic_vector(rec: Dict[str, Any], cache: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    topics = _record_topic_texts(
        rec,
        turns_per_topic=1,
        max_words_per_topic=60,
        min_words_informative=20,
        min_unique_words_informative=8,
    )
    vecs = []
    for item in topics.values():
        if not item.get("informative"):
            continue
        key = hashlib.md5(f"{DEFAULT_EMBEDDING_MODEL}\n{item['text']}".encode("utf-8")).hexdigest()
        vec = cache.get(key)
        if vec is not None:
            vecs.append(vec)
    if not vecs:
        return None
    mean_vec = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
    norm = np.linalg.norm(mean_vec)
    return mean_vec / norm if norm > 0 else mean_vec


def behavior_tokens(
    rec: Dict[str, Any],
    key_to_entry: Dict[str, Tuple[Dict[str, Dict[str, Set[str]]], Set[str]]],
) -> Optional[Set[str]]:
    entry = key_to_entry.get(record_cache_key(rec))
    if entry is None:
        return None
    extracted, tset = entry
    tokens: Set[str] = set()
    for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items():
        if topic_key not in tset:
            continue
        for attr in spec["attributes"]:
            for value in extracted[topic_key].get(attr, set()):
                tokens.add(f"{topic_key}::{attr}::{value}")
    return tokens or None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, min(2.0, 1.0 - float(np.dot(a, b))))


def jaccard_distance(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return 1.0 - (len(a & b) / len(union)) if union else 0.0


def distribution_panel(
    ax,
    same_values: Sequence[float],
    different_values: Sequence[float],
    title: str,
    xlabel: str,
    auc: float,
    effect_size: float,
) -> None:
    positions = [1, 0]

    violin = ax.violinplot(
        [same_values, different_values],
        positions=positions,
        vert=False,
        widths=0.65,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, color in zip(violin["bodies"], [COLORS["same"], COLORS["different"]]):
        body.set_alpha(0.30)
        body.set_facecolor(color)
        body.set_edgecolor("none")

    bp = ax.boxplot(
        [same_values, different_values],
        positions=positions,
        vert=False,
        widths=0.16,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "0.15", "linewidth": 1.1},
        whiskerprops={"linewidth": 0.7, "color": "0.25"},
        capprops={"linewidth": 0.6, "color": "0.25"},
        boxprops={"linewidth": 0.8, "edgecolor": "0.25"},
    )
    for patch, color in zip(bp["boxes"], [COLORS["same"], COLORS["different"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_yticks(positions)
    ax.set_yticklabels(["Same", "Different"])
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="semibold", pad=6)
    ax.text(
        0.98,
        0.93,
        f"AUC = {auc:.2f}, d = {effect_size:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.6, color="0.85")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def dumbbell_panel(
    ax,
    stages: Sequence[str],
    same_scores: Sequence[float],
    different_scores: Sequence[float],
    title: str,
    xlabel: str,
    auc: float,
    effect_size: float,
    xlim: Optional[Tuple[float, float]] = None,
    behavior_ticks: bool = False,
    stats_xy: Tuple[float, float] = (0.98, 0.03),
    stats_ha: str = "right",
    stats_va: str = "bottom",
    invert_y: bool = True,
    y_positions: Optional[Sequence[float]] = None,
) -> None:
    y = np.asarray(y_positions if y_positions is not None else np.arange(len(stages)), dtype=float)

    for yi, same, different in zip(y, same_scores, different_scores):
        ax.plot([same, different], [yi, yi], color="0.75", linewidth=1.0, zorder=1)

    ax.scatter(
        same_scores,
        y,
        s=24,
        color=COLORS["same"],
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )
    ax.scatter(
        different_scores,
        y,
        s=24,
        color=COLORS["different"],
        edgecolor="white",
        linewidth=0.4,
        zorder=3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(stages)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="semibold", pad=6)
    ax.text(
        stats_xy[0],
        stats_xy[1],
        f"AUC = {auc:.2f}, d = {effect_size:.2f}",
        transform=ax.transAxes,
        ha=stats_ha,
        va=stats_va,
        fontsize=7.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.2},
    )

    step = float(np.median(np.diff(np.sort(y)))) if len(y) > 1 else 1.0
    ax.set_ylim(float(y.min() - 0.55 * step), float(y.max() + 0.55 * step))
    if invert_y:
        ax.invert_yaxis()

    if xlim is not None:
        ax.set_xlim(xlim)
    if behavior_ticks:
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    ax.grid(axis="x", linestyle=":", linewidth=0.6, color="0.85")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def heatmap_panel(
    ax,
    matrix: np.ndarray,
    profile_boundaries: Sequence[int],
    profile_centers: Sequence[float],
    profile_labels: Sequence[str],
    title: Optional[str] = None,
) -> Any:
    im = ax.imshow(
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=np.nanmax(matrix),
        interpolation="nearest",
        aspect="equal",
    )

    for boundary in profile_boundaries:
        ax.axhline(boundary - 0.5, color="white", linewidth=0.9, alpha=0.9)
        ax.axvline(boundary - 0.5, color="white", linewidth=0.9, alpha=0.9)

    ax.set_xticks(profile_centers)
    ax.set_yticks(profile_centers)
    ax.set_xticklabels(profile_labels)
    ax.set_yticklabels(profile_labels)
    ax.tick_params(length=0, labelsize=8)
    if title:
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="semibold", pad=6)

    for spine in ax.spines.values():
        spine.set_visible(False)

    return im


def flatten_values(score_map: Dict[str, List[float]], topics: Sequence[str]) -> List[float]:
    out: List[float] = []
    for topic in topics:
        out.extend(score_map.get(topic, []))
    return out


def topic_means(
    within_map: Dict[str, List[float]],
    between_map: Dict[str, List[float]],
    topics: Optional[Sequence[str]] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    topic_order = topics if topics is not None else TOPIC_ORDER
    topic_keys: List[str] = []
    within_mean: List[float] = []
    between_mean: List[float] = []
    for topic in topic_order:
        w = within_map.get(topic, [])
        b = between_map.get(topic, [])
        if not w or not b:
            continue
        topic_keys.append(topic)
        within_mean.append(float(np.mean(np.asarray(w, dtype=float))))
        between_mean.append(float(np.mean(np.asarray(b, dtype=float))))
    return topic_keys, np.asarray(within_mean), np.asarray(between_mean)


def add_stage_separators(ax, y_positions: Sequence[float], separator_indices: Sequence[int]) -> None:
    y_arr = np.asarray(y_positions, dtype=float)
    for idx in separator_indices:
        if idx + 1 >= len(y_arr):
            continue
        separator_y = (y_arr[idx] + y_arr[idx + 1]) / 2.0
        ax.axhline(separator_y, color="0.88", linewidth=0.7, zorder=0)


def add_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=layout.RESULTS_DIR / "patient_psi_agenda_runs25.jsonl",
    )
    ap.add_argument("--record-cache", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=layout.VALIDITY_FINAL_DIR)
    ap.add_argument("--embed-cache", type=Path, default=layout.VALIDITY_DIR / "embed_cache.npz")
    ap.add_argument("--figure1-name", type=str, default="score_stage_validity")
    ap.add_argument("--figure2-name", type=str, default="conversation_level_distance")

    ap.add_argument("--semantic-metric", choices=["semantic", "group", "simulation"], default="simulation")
    ap.add_argument("--heatmap-metric", choices=["semantic", "behavior"], default="semantic")

    ap.add_argument("--group-size-score", type=int, default=10)
    ap.add_argument("--n-samples-score", type=int, default=300)
    ap.add_argument("--group-size-stage", type=int, default=10)
    ap.add_argument("--n-samples-stage", type=int, default=200)
    ap.add_argument("--n-profiles", type=int, default=8)
    ap.add_argument("--runs-per-profile", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)

    ap.add_argument("--figure1-width", type=float, default=7.2)
    ap.add_argument("--figure1-height", type=float, default=4.8)
    ap.add_argument("--figure2-width", type=float, default=3.0)
    ap.add_argument("--figure2-height", type=float, default=2.7)
    ap.add_argument(
        "--behavior-stage-xlim",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Optional fixed x-limits for behavior stage-level dumbbell panel.",
    )
    ap.add_argument(
        "--dump-scores",
        type=Path,
        default=None,
        help="Also write the score-level same/different-profile samples and their "
        "AUC / Cohen's d to this JSON (read by the metric-analysis figure).",
    )
    ap.add_argument(
        "--dump-stages",
        type=Path,
        default=None,
        help="Also write the stage-level same/different-profile means to this JSON "
        "(read by the metric-appendix figure).",
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"[load] {args.input}", flush=True)
    records = load_records(args.input)
    if not records:
        raise RuntimeError(f"No records in input: {args.input}")

    cache: Dict[str, np.ndarray] = {}
    if args.embed_cache.exists():
        npz = np.load(args.embed_cache, allow_pickle=False)
        cache = {k: npz[k] for k in npz.files}
    if not cache:
        raise RuntimeError(f"Embedding cache is empty or missing: {args.embed_cache}")

    rc_path = args.record_cache or Path(str(args.input) + ".record_cache.json")
    behavior_cache = load_record_cache(rc_path).get("behavior_extraction", {})
    key_to_entry: Dict[str, Tuple[Dict[str, Dict[str, Set[str]]], Set[str]]] = {}
    for rec in records:
        entry = behavior_cache.get(record_cache_key(rec))
        if isinstance(entry, dict):
            key_to_entry[record_cache_key(rec)] = (
                _deserialize_extracted_attributes(entry.get("extracted")),
                {str(t) for t in (entry.get("topics_with_patient_text") or []) if isinstance(t, str)},
            )
    print(f"[load] behavior cache entries={len(key_to_entry)}", flush=True)

    by_profile: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_profile[rec.get("profile_id")].append(rec)

    needed = max(args.group_size_score, args.group_size_stage)
    profiles = [p for p, runs in by_profile.items() if len(runs) >= needed]
    if len(profiles) < needed:
        raise RuntimeError(
            f"Need at least {needed} profiles with >= {needed} runs; got {len(profiles)} eligible profiles"
        )
    print(f"[load] records={len(records)} eligible_profiles={len(profiles)}", flush=True)

    semantic_fn = SEMANTIC_SCORE_FUNCS[args.semantic_metric]

    # ------------------------------------------------------------------
    # Score-level data (top row)
    # ------------------------------------------------------------------
    K_score = args.group_size_score
    within_score_groups = [rng.sample(by_profile[rng.choice(profiles)], K_score) for _ in range(args.n_samples_score)]
    between_score_groups = []
    for _ in range(args.n_samples_score):
        chosen = rng.sample(profiles, K_score)
        between_score_groups.append([rng.choice(by_profile[pid]) for pid in chosen])

    semantic_same = [
        value
        for value in (semantic_score(semantic_fn, group, cache) for group in within_score_groups)
        if value is not None
    ]
    semantic_diff = [
        value
        for value in (
            semantic_score(semantic_fn, relabel(group, f"MIX_{i}"), cache)
            for i, group in enumerate(between_score_groups)
        )
        if value is not None
    ]
    behavior_same = [
        value
        for value in (behavior_value_score(group, key_to_entry) for group in within_score_groups)
        if value is not None
    ]
    behavior_diff = [
        value
        for value in (behavior_value_score(group, key_to_entry) for group in between_score_groups)
        if value is not None
    ]
    if len(semantic_same) < 2 or len(semantic_diff) < 2:
        raise RuntimeError("Not enough semantic score samples to draw distributions")
    if len(behavior_same) < 2 or len(behavior_diff) < 2:
        raise RuntimeError("Not enough behavior score samples to draw distributions")

    sem_score_stats = summarize(semantic_same, semantic_diff)
    beh_score_stats = summarize(behavior_same, behavior_diff)
    print(
        "[score] semantic "
        f"AUC={sem_score_stats['auc']:.3f} d={sem_score_stats['cohens_d']:.3f} "
        f"({SEMANTIC_NAME[args.semantic_metric]})",
        flush=True,
    )
    print(
        "[score] behavior "
        f"AUC={beh_score_stats['auc']:.3f} d={beh_score_stats['cohens_d']:.3f}",
        flush=True,
    )
    if args.dump_scores is not None:
        args.dump_scores.parent.mkdir(parents=True, exist_ok=True)
        args.dump_scores.write_text(json.dumps({
            "semantic_metric": args.semantic_metric,
            "seed": args.seed,
            "simulation": {"same": semantic_same, "different": semantic_diff,
                           "stats": sem_score_stats},
            "behavior": {"same": behavior_same, "different": behavior_diff,
                         "stats": beh_score_stats},
        }, indent=1))
        print(f"[score] wrote {args.dump_scores}", flush=True)

    # ------------------------------------------------------------------
    # Stage-level data (middle row)
    # ------------------------------------------------------------------
    K_stage = args.group_size_stage
    within_stage_groups = [rng.sample(by_profile[rng.choice(profiles)], K_stage) for _ in range(args.n_samples_stage)]
    between_stage_groups = []
    for _ in range(args.n_samples_stage):
        chosen = rng.sample(profiles, K_stage)
        between_stage_groups.append([rng.choice(by_profile[pid]) for pid in chosen])

    semantic_stage_within: Dict[str, List[float]] = defaultdict(list)
    semantic_stage_between: Dict[str, List[float]] = defaultdict(list)
    behavior_stage_within: Dict[str, List[float]] = defaultdict(list)
    behavior_stage_between: Dict[str, List[float]] = defaultdict(list)

    for group in within_stage_groups:
        for topic_key, value in semantic_per_topic(semantic_fn, group, cache).items():
            semantic_stage_within[topic_key].append(value)
        for topic_key, value in behavior_per_topic(group, key_to_entry).items():
            behavior_stage_within[topic_key].append(value)

    for i, group in enumerate(between_stage_groups):
        relabeled = relabel(group, f"MIX_{i}")
        for topic_key, value in semantic_per_topic(semantic_fn, relabeled, cache).items():
            semantic_stage_between[topic_key].append(value)
        for topic_key, value in behavior_per_topic(group, key_to_entry).items():
            behavior_stage_between[topic_key].append(value)

    stage_topics = [
        topic
        for topic in TOPIC_ORDER
        if semantic_stage_within.get(topic)
        and semantic_stage_between.get(topic)
        and behavior_stage_within.get(topic)
        and behavior_stage_between.get(topic)
    ]
    if not stage_topics:
        raise RuntimeError("No shared stage-level topics had data across both metrics")

    _, sem_stage_same, sem_stage_diff = topic_means(
        semantic_stage_within,
        semantic_stage_between,
        topics=stage_topics,
    )
    _, beh_stage_same, beh_stage_diff = topic_means(
        behavior_stage_within,
        behavior_stage_between,
        topics=stage_topics,
    )

    sem_stage_stats = summarize(
        flatten_values(semantic_stage_within, stage_topics),
        flatten_values(semantic_stage_between, stage_topics),
    )
    beh_stage_stats = summarize(
        flatten_values(behavior_stage_within, stage_topics),
        flatten_values(behavior_stage_between, stage_topics),
    )
    print(
        f"[stage] semantic AUC={sem_stage_stats['auc']:.3f} d={sem_stage_stats['cohens_d']:.3f}",
        flush=True,
    )
    print(
        f"[stage] behavior AUC={beh_stage_stats['auc']:.3f} d={beh_stage_stats['cohens_d']:.3f}",
        flush=True,
    )

    if args.dump_stages is not None:
        args.dump_stages.parent.mkdir(parents=True, exist_ok=True)
        args.dump_stages.write_text(json.dumps({
            "semantic_metric": args.semantic_metric,
            "seed": args.seed,
            "stages": [TOPIC_DISPLAY[t] for t in stage_topics],
            "simulation": {"same": sem_stage_same.tolist(), "different": sem_stage_diff.tolist(),
                           "stats": sem_stage_stats},
            "behavior": {"same": beh_stage_same.tolist(), "different": beh_stage_diff.tolist(),
                         "stats": beh_stage_stats},
        }, indent=1))
        print(f"[stage] wrote {args.dump_stages}", flush=True)

    # ------------------------------------------------------------------
    # Heatmap data (bottom row)
    # ------------------------------------------------------------------
    by_profile_vectors: Dict[Any, List[Tuple[np.ndarray, Set[str]]]] = defaultdict(list)
    for rec in records:
        sem_vec = semantic_vector(rec, cache)
        beh_tok = behavior_tokens(rec, key_to_entry)
        if sem_vec is not None and beh_tok is not None:
            by_profile_vectors[rec.get("profile_id")].append((sem_vec, beh_tok))

    eligible_profiles = sorted(
        [profile for profile, values in by_profile_vectors.items() if len(values) >= args.runs_per_profile],
        key=str,
    )[: args.n_profiles]
    if len(eligible_profiles) < 2:
        raise RuntimeError("Not enough profiles with semantic+behavior vectors for heatmap")

    sem_vecs: List[np.ndarray] = []
    beh_sets: List[Set[str]] = []
    boundaries: List[int] = []
    for profile in eligible_profiles:
        items = rng.sample(by_profile_vectors[profile], args.runs_per_profile)
        for sem_vec, beh_tok in items:
            sem_vecs.append(sem_vec)
            beh_sets.append(beh_tok)
        boundaries.append(len(sem_vecs))

    n = len(sem_vecs)
    sem_mat = np.zeros((n, n), dtype=np.float32)
    beh_mat = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            sem_mat[i, j] = cosine_distance(sem_vecs[i], sem_vecs[j])
            beh_mat[i, j] = jaccard_distance(beh_sets[i], beh_sets[j])

    heatmap_mat = sem_mat if args.heatmap_metric == "semantic" else beh_mat
    heatmap_cbar_label = "Cosine distance" if args.heatmap_metric == "semantic" else "Jaccard distance"

    centers: List[float] = []
    prev = 0
    for boundary in boundaries:
        centers.append((prev + boundary - 1) / 2.0)
        prev = boundary
    profile_labels = [f"P{i}" for i in range(len(eligible_profiles))]

    # ------------------------------------------------------------------
    # Figure 1: score-level + stage-level validity (2x2)
    # ------------------------------------------------------------------
    add_rcparams()
    fig1 = plt.figure(figsize=(args.figure1_width, args.figure1_height))
    fig1.patch.set_facecolor("white")
    gs = fig1.add_gridspec(
        2,
        2,
        height_ratios=[0.75, 1.25],
        hspace=0.55,
        wspace=0.18,
    )

    ax_a = fig1.add_subplot(gs[0, 0])
    ax_b = fig1.add_subplot(gs[0, 1])
    ax_c = fig1.add_subplot(gs[1, 0])
    ax_d = fig1.add_subplot(gs[1, 1], sharey=ax_c)

    distribution_panel(
        ax_a,
        semantic_same,
        semantic_diff,
        "(a) Semantic score-level validity",
        "Diversity score",
        auc=sem_score_stats["auc"],
        effect_size=sem_score_stats["cohens_d"],
    )
    distribution_panel(
        ax_b,
        behavior_same,
        behavior_diff,
        "(b) Behavior score-level validity",
        "Diversity score",
        auc=beh_score_stats["auc"],
        effect_size=beh_score_stats["cohens_d"],
    )

    stage_labels = [TOPIC_DISPLAY[t] for t in stage_topics]
    y_positions = np.arange(len(stage_labels), dtype=float) * 0.82

    dumbbell_panel(
        ax_c,
        stage_labels,
        sem_stage_same,
        sem_stage_diff,
        "(c) Semantic validity by stage",
        "Diversity score",
        auc=sem_stage_stats["auc"],
        effect_size=sem_stage_stats["cohens_d"],
        stats_xy=(0.98, 0.03),
        stats_ha="right",
        stats_va="bottom",
        y_positions=y_positions,
    )

    if args.behavior_stage_xlim is not None:
        behavior_xlim = (float(args.behavior_stage_xlim[0]), float(args.behavior_stage_xlim[1]))
    else:
        lo = float(min(np.min(beh_stage_same), np.min(beh_stage_diff)))
        hi = float(max(np.max(beh_stage_same), np.max(beh_stage_diff)))
        pad = max(0.0008, 0.08 * (hi - lo))
        behavior_xlim = (max(0.0, lo - pad), min(1.0, hi + pad))

    dumbbell_panel(
        ax_d,
        stage_labels,
        beh_stage_same,
        beh_stage_diff,
        "(d) Behavior validity by stage",
        "Diversity score",
        auc=beh_stage_stats["auc"],
        effect_size=beh_stage_stats["cohens_d"],
        xlim=behavior_xlim,
        behavior_ticks=True,
        stats_xy=(0.03, 0.97),
        stats_ha="left",
        stats_va="top",
        y_positions=y_positions,
    )

    add_stage_separators(ax_c, y_positions, STAGE_SEPARATOR_INDICES)
    add_stage_separators(ax_d, y_positions, STAGE_SEPARATOR_INDICES)

    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels(stage_labels, fontsize=7.5)
    ax_d.set_yticks(y_positions)
    ax_d.set_ylim(ax_c.get_ylim())
    plt.setp(ax_d.get_yticklabels(), visible=False)
    ax_d.tick_params(axis="y", length=0)

    for ax in [ax_a, ax_b, ax_c, ax_d]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", linewidth=0.6, color="0.85")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["same"],
            markeredgecolor="none",
            markersize=6,
            label="Same profile",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=COLORS["different"],
            markeredgecolor="none",
            markersize=6,
            label="Different profile",
        ),
    ]
    fig1.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        fontsize=8,
        handletextpad=0.4,
        columnspacing=1.4,
    )
    fig1.subplots_adjust(top=0.90)

    fig1_pdf = args.outdir / f"{args.figure1_name}.pdf"
    fig1_png = args.outdir / f"{args.figure1_name}.png"
    fig1.savefig(fig1_pdf, bbox_inches="tight", pad_inches=0.02)
    fig1.savefig(fig1_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig1)
    print(f"[write] {fig1_pdf}", flush=True)
    print(f"[write] {fig1_png}", flush=True)

    # ------------------------------------------------------------------
    # Figure 2: conversation-level distance heatmap (single-column)
    # ------------------------------------------------------------------
    fig2, ax_hm = plt.subplots(figsize=(args.figure2_width, args.figure2_height))
    fig2.patch.set_facecolor("white")
    im = heatmap_panel(
        ax_hm,
        heatmap_mat,
        boundaries[:-1],
        centers,
        profile_labels,
        title=None,
    )
    cbar = fig2.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.035)
    cbar.set_label(heatmap_cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig2_pdf = args.outdir / f"{args.figure2_name}.pdf"
    fig2_png = args.outdir / f"{args.figure2_name}.png"
    fig2.savefig(fig2_pdf, bbox_inches="tight", pad_inches=0.02)
    fig2.savefig(fig2_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig2)
    print(f"[write] {fig2_pdf}", flush=True)
    print(f"[write] {fig2_png}", flush=True)


if __name__ == "__main__":
    main()
