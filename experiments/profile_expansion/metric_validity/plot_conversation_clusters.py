#!/usr/bin/env python3
"""2D cluster view of conversation diversity.

Each dot is one conversation (one run). Its position comes from a 2D projection
(t-SNE / PCA) so the distance between dots reflects how different the
conversations are. Dots are colored by source profile.

Validity reading: if the diversity distance is meaningful, conversations from
the SAME short profile form a tight cluster and clusters from DIFFERENT
profiles sit apart. The cosine/Jaccard silhouette (profile = label, computed in
the ORIGINAL feature space, not the 2D view) quantifies this: +1 = cleanly
separated, 0 = overlapping, <0 = mixed.

Spaces (``--space``)
  semantic        cosine on the mean per-topic embedding (semantic_diversity).
  behavior_combo  Jaccard on the set of mentioned (topic, attribute) pairs.
  behavior_value  Jaccard on the set of (topic, attribute, value) triples.

Reuses on-disk caches: the embedding cache for semantic, the behavior-extraction
record cache for behavior -- no recompute, no LLM calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from angel_common.paths import REPO_ROOT

PACKAGE_ROOT = REPO_ROOT  # repository root; run scripts with `python -m` from here

from experiments.profile_expansion.metrics.semantic_diversity import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    _record_topic_texts,
)
from experiments.profile_expansion.metrics.behavior_diversity import (  # noqa: E402
    TOPIC_ATTRIBUTE_SCHEMA,
    _deserialize_extracted_attributes,
)
from experiments.profile_expansion.evaluate_metrics_record_cache import (  # noqa: E402
    load_record_cache,
    record_cache_key,
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Per-conversation feature builders
# ---------------------------------------------------------------------------
def build_semantic_vectors(
    records: List[Dict[str, Any]], *, embedding_cache: Dict[str, np.ndarray]
) -> Tuple[List[Any], np.ndarray]:
    """One vector per conversation = L2-normalized mean of its per-topic
    informative embeddings (the representation semantic_diversity scores on).

    Reads vectors straight from the on-disk cache (same md5 key the metric uses);
    any text not already cached is skipped, so the embedding model is never
    loaded here."""

    def lookup(text: str) -> Optional[np.ndarray]:
        key = hashlib.md5(f"{DEFAULT_EMBEDDING_MODEL}\n{text}".encode("utf-8")).hexdigest()
        return embedding_cache.get(key)

    profile_ids: List[Any] = []
    rows: List[np.ndarray] = []
    n_missing = 0
    for rec in records:
        topics = _record_topic_texts(
            rec, turns_per_topic=1, max_words_per_topic=60,
            min_words_informative=20, min_unique_words_informative=8,
        )
        texts = [str(item["text"]) for item in topics.values() if item.get("informative")]
        vecs = []
        for t in texts:
            v = lookup(t)
            if v is None:
                n_missing += 1
            else:
                vecs.append(v)
        if not vecs:
            continue
        profile_id = rec.get("profile_id")
        mean_vec = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        profile_ids.append(profile_id)
        rows.append(mean_vec)
    return profile_ids, np.asarray(rows, dtype=np.float32)


def build_behavior_features(
    records: List[Dict[str, Any]], *, record_cache: Dict[str, Any], level: str
) -> Tuple[List[Any], np.ndarray]:
    """One binary feature vector per conversation over a behavior vocabulary.

    level="combo": tokens are "topic::attr" for each mentioned attribute.
    level="value": tokens are "topic::attr::value" for each extracted value.
    Jaccard distance on these binary vectors is the behavior diversity signal.
    """
    behavior_cache = record_cache.get("behavior_extraction", {})
    per_record: List[Tuple[Any, set]] = []
    vocab: Dict[str, int] = {}

    for rec in records:
        entry = behavior_cache.get(record_cache_key(rec))
        if not isinstance(entry, dict):
            continue
        extracted = _deserialize_extracted_attributes(entry.get("extracted"))
        topics_with_text = {
            str(t) for t in (entry.get("topics_with_patient_text") or []) if isinstance(t, str)
        }
        tokens: set = set()
        for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items():
            if topic_key not in topics_with_text:
                continue
            topic_values = extracted.get(topic_key, {})
            for attr in spec["attributes"]:
                vals = topic_values.get(attr)
                if not vals:
                    continue
                if level == "combo":
                    tokens.add(f"{topic_key}::{attr}")
                else:
                    for v in vals:
                        tokens.add(f"{topic_key}::{attr}::{v}")
        if not tokens:
            continue
        for tok in tokens:
            if tok not in vocab:
                vocab[tok] = len(vocab)
        per_record.append((rec.get("profile_id"), tokens))

    profile_ids: List[Any] = []
    mat = np.zeros((len(per_record), len(vocab)), dtype=np.float32)
    for i, (profile_id, tokens) in enumerate(per_record):
        profile_ids.append(profile_id)
        for tok in tokens:
            mat[i, vocab[tok]] = 1.0
    return profile_ids, mat


# ---------------------------------------------------------------------------
# Subsample / project
# ---------------------------------------------------------------------------
def subsample(profile_ids, matrix, *, n_profiles, runs_per_profile, rng):
    by_profile: Dict[Any, List[int]] = defaultdict(list)
    for idx, pid in enumerate(profile_ids):
        by_profile[pid].append(idx)
    eligible = sorted(by_profile.keys(), key=lambda p: (-len(by_profile[p]), str(p)))
    chosen = eligible[:n_profiles] if n_profiles > 0 else eligible

    keep_ids, keep_rows = [], []
    for pid in chosen:
        idxs = by_profile[pid]
        if runs_per_profile > 0 and len(idxs) > runs_per_profile:
            idxs = sorted(rng.sample(idxs, runs_per_profile))
        for i in idxs:
            keep_ids.append(pid)
            keep_rows.append(matrix[i])
    return keep_ids, np.asarray(keep_rows, dtype=np.float32)


def project(matrix: np.ndarray, *, method: str, metric: str, seed: int) -> np.ndarray:
    from sklearn.decomposition import PCA

    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(matrix)

    from sklearn.manifold import TSNE

    n = matrix.shape[0]
    perplexity = max(5, min(30, (n - 1) // 3))
    if metric == "cosine":
        pre = PCA(n_components=min(50, matrix.shape[1]), random_state=seed).fit_transform(matrix)
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                    learning_rate="auto", random_state=seed, metric="cosine")
        return tsne.fit_transform(pre)
    # Jaccard on binary features: project directly, random init.
    tsne = TSNE(n_components=2, perplexity=perplexity, init="random",
                learning_rate="auto", random_state=seed, metric=metric)
    return tsne.fit_transform(matrix)


# ---------------------------------------------------------------------------
# Plot (no centroid labels, no ellipses)
# ---------------------------------------------------------------------------
def plot(profile_ids, coords, *, method, metric, silhouette, n_profiles_shown,
         outpath: Path, space_label: str) -> None:
    uniq = sorted(set(profile_ids), key=str)
    cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
    color_of = {pid: cmap(i % cmap.N) for i, pid in enumerate(uniq)}
    # Sequential display label per profile: "profile 0", "profile 1", ...
    label_of = {pid: f"profile {i}" for i, pid in enumerate(uniq)}
    pid_arr = np.asarray(profile_ids, dtype=object)
    show_legend = len(uniq) <= 20

    fig, ax = plt.subplots(figsize=(10.5 if show_legend else 9.5, 8.5))
    for pid in uniq:
        pts = coords[pid_arr == pid]
        ax.scatter(pts[:, 0], pts[:, 1], s=34, color=color_of[pid],
                   edgecolor="white", linewidth=0.3, alpha=0.9,
                   label=label_of[pid] if show_legend else None)

    sil_str = f"   silhouette={silhouette:.3f}" if silhouette is not None else ""
    ax.set_title(
        f"Conversation clusters by profile -- {space_label} ({method.upper()})\n"
        f"each dot = one conversation; closer = more similar   "
        f"[{n_profiles_shown} profiles]{sil_str}",
        fontsize=12,
    )
    ax.set_xlabel(f"{method.upper()} dim 1")
    ax.set_ylabel(f"{method.upper()} dim 2")
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
                  ncol=1, title="profile", frameon=False)
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
    ap.add_argument("--record-cache", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=layout.VALIDITY_DIR)
    ap.add_argument("--embed-cache", type=Path, default=layout.VALIDITY_DIR / "embed_cache.npz")
    ap.add_argument("--space", choices=["semantic", "behavior_combo", "behavior_value"],
                    default="semantic")
    ap.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    ap.add_argument("--n-profiles", type=int, default=10, help="0 = all profiles")
    ap.add_argument("--runs-per-profile", type=int, default=20, help="0 = all runs")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"[load] {args.input}  space={args.space}", flush=True)
    records = load_records(args.input)

    if args.space == "semantic":
        embedding_cache: Dict[str, np.ndarray] = {}
        if args.embed_cache.exists():
            npz = np.load(args.embed_cache, allow_pickle=False)
            embedding_cache = {k: npz[k] for k in npz.files}
        profile_ids, matrix = build_semantic_vectors(records, embedding_cache=embedding_cache)
        np.savez(args.embed_cache, **embedding_cache)
        metric = "cosine"
        space_label = "semantic (cosine)"
    else:
        cache_path = args.record_cache or Path(str(args.input) + ".record_cache.json")
        record_cache = load_record_cache(cache_path)
        level = "combo" if args.space == "behavior_combo" else "value"
        profile_ids, matrix = build_behavior_features(records, record_cache=record_cache, level=level)
        metric = "jaccard"
        space_label = f"behavior {level} (jaccard)"

    print(f"[vectors] {matrix.shape[0]} conversations, dim={matrix.shape[1]}", flush=True)

    sub_ids, sub_mat = subsample(profile_ids, matrix, n_profiles=args.n_profiles,
                                 runs_per_profile=args.runs_per_profile, rng=rng)
    n_shown = len(set(sub_ids))
    print(f"[subsample] {sub_mat.shape[0]} conversations across {n_shown} profiles", flush=True)

    coords = project(sub_mat, method=args.method, metric=metric, seed=args.seed)

    silhouette = None
    if n_shown > 1:
        from sklearn.metrics import silhouette_score
        silhouette = float(silhouette_score(sub_mat, sub_ids, metric=metric))
        print(f"[silhouette] {metric}={silhouette:.4f}", flush=True)

    scope = "all" if args.n_profiles == 0 else f"n{args.n_profiles}"
    outpath = args.outdir / f"clusters_{args.space}_{args.method}_{scope}.png"
    plot(sub_ids, coords, method=args.method, metric=metric, silhouette=silhouette,
         n_profiles_shown=n_shown, outpath=outpath, space_label=space_label)


if __name__ == "__main__":
    main()
