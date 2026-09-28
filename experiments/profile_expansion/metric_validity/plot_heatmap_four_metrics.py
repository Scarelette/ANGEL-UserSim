#!/usr/bin/env python3
"""Per-conversation distance heatmaps for four diversity metrics (redraw fig3).

Each panel is a conversation x conversation distance matrix over the same set of
conversations, ordered/blocked by profile. A valid metric shows a dark
(low-distance) block diagonal: runs of one profile are close to each other and
far from other profiles.

NOTE: semantic / group / simulation all use the SAME pairwise cosine distance --
they differ only in how they aggregate across many runs, which does not apply to
a single pair -- so those three panels are identical by construction. Only
Behavior (Jaccard over extracted value tokens) is a distinct matrix.

Conversation distances:
  semantic family : cosine on the L2-normalized mean per-topic embedding.
  behavior        : Jaccard over the union of (topic, attribute, value) tokens.
Reuses the embedding + behavior extraction caches; no LLM calls.
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
from typing import Any, Dict, List, Optional, Set, Tuple

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

# Shared ACL figure style (lives with the results figures).
from experiments.profile_expansion.figures import acl_fig_style as st  # noqa: E402


def load_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def semantic_vector(rec: Dict[str, Any], cache: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    topics = _record_topic_texts(rec, turns_per_topic=1, max_words_per_topic=60,
                                 min_words_informative=20, min_unique_words_informative=8)
    vecs = []
    for item in topics.values():
        if not item.get("informative"):
            continue
        key = hashlib.md5(f"{DEFAULT_EMBEDDING_MODEL}\n{item['text']}".encode("utf-8")).hexdigest()
        v = cache.get(key)
        if v is not None:
            vecs.append(v)
    if not vecs:
        return None
    m = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
    n = np.linalg.norm(m)
    return m / n if n > 0 else m


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
            for v in extracted[topic_key].get(attr, set()):
                tokens.add(f"{topic_key}::{attr}::{v}")
    return tokens or None


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return max(0.0, min(2.0, 1.0 - float(np.dot(a, b))))


def jaccard_distance(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    return 1.0 - (len(a & b) / len(u)) if u else 0.0


def draw_heatmap(ax, mat, boundaries, labels, title=None):
    """One distance matrix. `title` is accepted but not drawn: the figures carry
    no in-figure titles, the LaTeX caption names the metric."""
    im = ax.imshow(mat, cmap="viridis", vmin=0.0)
    prev, centers, edges = 0, [], []
    for b in boundaries:
        centers.append((prev + b - 1) / 2.0)
        edges.append(b - 0.5)
        prev = b
    for e in edges[:-1]:
        ax.axhline(e, color="white", lw=0.45 * st.SCALE)
        ax.axvline(e, color="white", lw=0.45 * st.SCALE)
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(labels)  # short enough to sit horizontally
    ax.set_yticklabels(labels)
    ax.tick_params(length=0.0)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    return im


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=layout.RESULTS_DIR / "patient_psi_agenda_runs25.jsonl")
    ap.add_argument("--record-cache", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=layout.VALIDITY_DIR)
    ap.add_argument("--embed-cache", type=Path, default=layout.VALIDITY_DIR / "embed_cache.npz")
    ap.add_argument("--panels", choices=["two", "three", "four"], default="four")
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        choices=["semantic", "group", "simulation", "behavior"],
        help="Explicit metric panels (in this order); overrides --panels.",
    )
    ap.add_argument("--n-profiles", type=int, default=8)
    ap.add_argument("--runs-per-profile", type=int, default=6)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--width", choices=["column", "text"], default="column",
                    help="Printed width: one ACL column, or the two-column span.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    records = load_records(args.input)
    cache: Dict[str, np.ndarray] = {}
    if args.embed_cache.exists():
        npz = np.load(args.embed_cache, allow_pickle=False)
        cache = {k: npz[k] for k in npz.files}

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

    # Keep only conversations that have BOTH a semantic vector and behavior tokens.
    by_profile: Dict[Any, List[Tuple[np.ndarray, Set[str]]]] = defaultdict(list)
    for rec in records:
        sv = semantic_vector(rec, cache)
        bt = behavior_tokens(rec, key_to_entry)
        if sv is not None and bt is not None:
            by_profile[rec.get("profile_id")].append((sv, bt))

    eligible = sorted([p for p, v in by_profile.items() if len(v) >= args.runs_per_profile],
                      key=str)[:args.n_profiles]
    if len(eligible) < 2:
        print("[heatmap] not enough profiles", flush=True)
        return

    sem_vecs: List[np.ndarray] = []
    beh_sets: List[Set[str]] = []
    boundaries: List[int] = []
    labels: List[str] = []
    for i, pid in enumerate(eligible):
        items = rng.sample(by_profile[pid], args.runs_per_profile)
        for sv, bt in items:
            sem_vecs.append(sv)
            beh_sets.append(bt)
        boundaries.append(len(sem_vecs))
        labels.append(f"P{i}")

    n = len(sem_vecs)
    sem_mat = np.zeros((n, n))
    beh_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sem_mat[i, j] = cosine_distance(sem_vecs[i], sem_vecs[j])
            beh_mat[i, j] = jaccard_distance(beh_sets[i], beh_sets[j])

    if args.metrics:
        title_mat = {
            "semantic": ("Semantic Diversity (cosine)", sem_mat),
            "group": ("Group Diversity (cosine)", sem_mat),
            "simulation": ("Simulation Diversity (cosine)", sem_mat),
            "behavior": ("Behavior Diversity (Jaccard)", beh_mat),
        }
        panels = [title_mat[name] for name in args.metrics]
        ncols = min(2, len(panels))
        nrows = (len(panels) + ncols - 1) // ncols
        st.apply_compact_style()
        width_in = st.width_for(args.width)
        # Square-ish cells: one panel is as tall as it is wide, minus the labels.
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=st.size(width_in, width_in * 0.95 * nrows / ncols),
            layout="constrained",
        )
        fig.get_layout_engine().set(w_pad=0.01 * st.SCALE, h_pad=0.01 * st.SCALE)
        top_frac = None
        out = args.outdir / "fig3_conversation_distance_heatmap_three_metrics.png"
    elif args.panels == "four":
        panels = [
            ("Semantic Diversity (cosine)", sem_mat),
            ("Group Diversity (cosine)", sem_mat),
            ("Simulation Diversity (cosine)", sem_mat),
            ("Behavior Diversity (Jaccard)", beh_mat),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(15, 13.5))
        out = args.outdir / "fig3_conversation_distance_heatmap_four_metrics.png"
    elif args.panels == "three":
        panels = [
            ("Semantic Diversity (cosine)", sem_mat),
            ("Group Diversity (cosine)", sem_mat),
            ("Simulation Diversity (cosine)", sem_mat),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(22, 7.2))
        out = args.outdir / "fig3_conversation_distance_heatmap_three_metrics.png"
    else:
        panels = [
            ("Semantic Diversity (cosine)\nshared by semantic / group / simulation", sem_mat),
            ("Behavior Diversity (Jaccard)", beh_mat),
        ]
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
        out = args.outdir / "fig3_conversation_distance_heatmap_two_metrics.png"

    for ax, (title, mat) in zip(np.atleast_1d(axes).ravel(), panels):
        im = draw_heatmap(ax, mat, boundaries, labels)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="Distance")
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=1.2 * st.SCALE, width=0.7 * st.SCALE)

    # No suptitle / panel titles: the caption carries them.
    if args.metrics:
        st.save(fig, out.with_suffix(".pdf"), out)
        print(f"[write] {out.with_suffix('.pdf')}", flush=True)
        return
    fig.tight_layout()
    fig.subplots_adjust(top=0.88 if args.panels == "two" else 0.92)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out}", flush=True)


if __name__ == "__main__":
    main()
