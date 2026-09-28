#!/usr/bin/env python3
"""Combined score-level validity figure (fig1 style) for all diversity metrics.

One row of panels, each comparing a metric's EXACT diversity score on
same-profile groups vs different-profile groups of K runs:

    semantic       score_semantic_diversity      (all-pairs cosine)
    group          score_group_diversity         (k-nearest cosine)
    simulation     score_min_distance_diversity  (nearest-neighbor cosine)
    behavior value behavior_diversity value component (per-attribute Jaccard
                   over extracted VALUE strings, from cached extractions)

A valid metric scores same-profile groups LOW and different-profile groups HIGH.
Semantic scorers reuse the embedding cache; behavior reuses the behavior
extraction record cache. No LLM calls.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
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
from experiments.profile_expansion.metrics.behavior_diversity import (  # noqa: E402
    TOPIC_ATTRIBUTE_SCHEMA,
    _deserialize_extracted_attributes,
    _jaccard_distance,
)
from experiments.profile_expansion.metrics.common import mean as _mean  # noqa: E402
from experiments.profile_expansion.evaluate_metrics_record_cache import (  # noqa: E402
    load_record_cache,
    record_cache_key,
)

import json  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

from experiments.profile_expansion.figures import acl_fig_style as st  # noqa: E402

# Same two hues the rest of the figure set uses (Angel blue / Roleplay-doh red),
# so "same-profile" and "different-profile" read consistently across the paper.
# Type ramp copied from results/fig/behavior_d/topic_behavior_d.pdf:
# 15.75 / 16.8 / 16.1 nominal pt on the SCALE=2 canvas.
# Type hierarchy: panel title > axis label > tick > legend/annotation.
TITLE_PT = 17.0 / 2.0
SUBTITLE_PT = 12.5 / 2.0
LABEL_PT = 15.5 / 2.0
TICK_PT = 13.5 / 2.0
LEGEND_PT = 16.0 / 2.0
ANNOT_PT = 12.5 / 2.0
MEANLAB_PT = 11.5 / 2.0
PANEL_HEIGHT = 3.15  # printed inches per panel row (matches fig2)

# Point offsets above the axes for the stacked header rows.
ROW_ARROW = 5.0
ROW_SUBTITLE = 15.0
ROW_TITLE = 30.0

# Muted blue / muted coral: the same hues as the bar figures, desaturated so the
# 45%-alpha overlap stays readable in print.
PALETTE = {"within": "#5E8FC4", "between": "#E3918A"}
# Darker versions for the thin mean lines and their labels.
INK_PALETTE = {"within": "#2F6098", "between": "#C25C52"}

HIST_ALPHA = 0.45

# Panel heading + italic subtitle, keyed by the internal metric title. The left
# panel is named "Semantic Diversity" to match the wording used elsewhere in the
# paper; the subtitle states the measure actually plotted.
PANEL_TITLES = {
    "Semantic Diversity (all-pairs)": ("Semantic Diversity", "All-pairs cosine similarity"),
    "Group Diversity (k-nearest)": ("Group Diversity", "k-nearest similarity"),
    "Simulation Diversity (nearest-neighbor)": ("Semantic Diversity", "Nearest-neighbor similarity"),
    "Behavior Diversity (value Jaccard)": ("Behavior Diversity", "Value Jaccard similarity"),
}

PANEL_XLABEL = {
    "Semantic Diversity (all-pairs)": "Semantic diversity",
    "Group Diversity (k-nearest)": "Group diversity",
    "Simulation Diversity (nearest-neighbor)": "Semantic diversity",
    "Behavior Diversity (value Jaccard)": "Behavior diversity",
}


def load_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# Semantic-family: exact scorers (group by model+profile_id)
# ---------------------------------------------------------------------------
def relabel(group: List[Dict[str, Any]], tag: str) -> List[Dict[str, Any]]:
    out = []
    for rec in group:
        r = dict(rec)
        r["profile_id"] = tag
        r["model"] = "MIX"
        out.append(r)
    return out


def semantic_score(score_fn: Callable, group: List[Dict[str, Any]],
                   cache: Dict[str, np.ndarray]) -> Optional[float]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = score_fn(group, embedding_cache=cache)
    s = result.get("score")
    return float(s) if s is not None else None


# ---------------------------------------------------------------------------
# Behavior value: per-attribute Jaccard over extracted value strings,
# replicating behavior_diversity's value-diversity component for a group.
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
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


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=layout.RESULTS_DIR / "patient_psi_agenda_runs25.jsonl")
    ap.add_argument("--record-cache", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=layout.VALIDITY_DIR)
    ap.add_argument("--embed-cache", type=Path, default=layout.VALIDITY_DIR / "embed_cache.npz")
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--width", choices=["column", "text"], default="text",
                    help="Printed width: one ACL column, or the two-column span.")
    ap.add_argument(
        "--metrics",
        nargs="+",
        default=["semantic", "group", "simulation", "behavior_value"],
        choices=["semantic", "group", "simulation", "behavior_value"],
        help="Which metric panels to render (in this order).",
    )
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    K = args.group_size

    print(f"[load] {args.input}", flush=True)
    records = load_records(args.input)
    by_profile: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_profile[rec.get("profile_id")].append(rec)
    profiles = [p for p, runs in by_profile.items() if len(runs) >= K]
    print(f"[load] {len(records)} records; {len(profiles)} profiles with >= {K} runs", flush=True)

    # Embedding cache (fill any gaps once).
    cache: Dict[str, np.ndarray] = {}
    if args.embed_cache.exists():
        npz = np.load(args.embed_cache, allow_pickle=False)
        cache = {k: npz[k] for k in npz.files}
    all_texts = set()
    for rec in records:
        topics = _record_topic_texts(rec, turns_per_topic=1, max_words_per_topic=60,
                                     min_words_informative=20, min_unique_words_informative=8)
        for item in topics.values():
            if item.get("informative"):
                all_texts.add(str(item["text"]))
    _build_embeddings(sorted(all_texts), embedding_model=DEFAULT_EMBEDDING_MODEL, embedding_cache=cache)
    np.savez(args.embed_cache, **cache)

    # Behavior extraction cache -> deserialized entries by record key.
    rc_path = args.record_cache or Path(str(args.input) + ".record_cache.json")
    record_cache = load_record_cache(rc_path)
    behavior_cache = record_cache.get("behavior_extraction", {})
    key_to_entry: Dict[str, Tuple[Dict[str, Dict[str, Set[str]]], Set[str]]] = {}
    for rec in records:
        key = record_cache_key(rec)
        entry = behavior_cache.get(key)
        if isinstance(entry, dict):
            extracted = _deserialize_extracted_attributes(entry.get("extracted"))
            tset = {str(t) for t in (entry.get("topics_with_patient_text") or []) if isinstance(t, str)}
            key_to_entry[key] = (extracted, tset)
    print(f"[behavior] {len(key_to_entry)} cached extractions", flush=True)

    # Shared group memberships (original records).
    within_groups = [rng.sample(by_profile[rng.choice(profiles)], K) for _ in range(args.n_samples)]
    between_groups = []
    for _ in range(args.n_samples):
        chosen = rng.sample(profiles, K)
        between_groups.append([rng.choice(by_profile[pid]) for pid in chosen])

    all_panels = [
        ("Semantic Diversity (all-pairs)", "semantic",
         lambda g: semantic_score(score_semantic_diversity, g, cache),
         lambda g, i: semantic_score(score_semantic_diversity, relabel(g, f"MIX_{i}"), cache)),
        ("Group Diversity (k-nearest)", "group",
         lambda g: semantic_score(score_group_diversity, g, cache),
         lambda g, i: semantic_score(score_group_diversity, relabel(g, f"MIX_{i}"), cache)),
        ("Simulation Diversity (nearest-neighbor)", "simulation",
         lambda g: semantic_score(score_min_distance_diversity, g, cache),
         lambda g, i: semantic_score(score_min_distance_diversity, relabel(g, f"MIX_{i}"), cache)),
        ("Behavior Diversity (value Jaccard)", "behavior_value",
         lambda g: behavior_value_score(g, key_to_entry),
         lambda g, i: behavior_value_score(g, key_to_entry)),
    ]
    by_name = {name: panel for panel in all_panels for name in (panel[1],)}
    panels = [by_name[name] for name in args.metrics]

    n_panels = len(panels)
    ncols = min(2, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    st.apply_style(tick=TICK_PT, label=LABEL_PT, legend=LEGEND_PT, title=TITLE_PT)
    S = st.SCALE
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=st.size(st.width_for(args.width), PANEL_HEIGHT * nrows),
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.015 * S, h_pad=0.04 * S, wspace=0.025)
    flat_axes = np.atleast_1d(axes).ravel()
    summary: Dict[str, Dict[str, float]] = {}
    handles_labels = None
    for ax, (title, name, within_fn, between_fn) in zip(flat_axes, panels):
        print(f"[score] {name} ...", flush=True)
        within = [v for v in (within_fn(g) for g in within_groups) if v is not None]
        between = [v for v in (between_fn(g, i) for i, g in enumerate(between_groups)) if v is not None]
        stats = summarize(within, between)
        summary[name] = stats
        print(f"   within={stats['mean_within']:.4f} between={stats['mean_between']:.4f} "
              f"AUC={stats['auc']:.3f} ratio={stats['separation_ratio']:.2f} [{verdict(stats)}]",
              flush=True)

        # Trim the empty left half: both distributions live well above zero.
        lo = min(min(within), min(between))
        hi = max(max(within), max(between)) + 1e-6
        span = hi - lo
        bins = np.linspace(max(0.0, lo - span * 0.25), hi + span * 0.05, 36)
        for values, color, label in (
            (within, PALETTE["within"], "Same profile"),
            (between, PALETTE["between"], "Different profile"),
        ):
            ax.hist(values, bins=bins, density=True, histtype="stepfilled",
                    facecolor=color, alpha=HIST_ALPHA, edgecolor=color,
                    linewidth=0.6 * S, label=label, zorder=3)

        # Thin dashed mean lines, each labelled with its group and value so the
        # reader never has to estimate the centres.
        for key, mean, name, ha, y in (
            # Both labels hang to the left of their line and are staggered in
            # height: to the right lies the statistics card.
            ("within", stats["mean_within"], "Same", "right", 0.930),
            ("between", stats["mean_between"], "Different", "right", 0.815),
        ):
            ax.axvline(mean, color=INK_PALETTE[key], ls=(0, (4, 3)),
                       lw=0.55 * S, zorder=5)
            ax.annotate(f"{name}  \u03bc={mean:.2f}",
                        xy=(mean, y), xycoords=("data", "axes fraction"),
                        xytext=(3 * S if ha == "left" else -3 * S, 0),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=st.pt(MEANLAB_PT), color=INK_PALETTE[key],
                        zorder=6)

        # No panel title/subtitle for the final export.

        # A subtle reading aid: which way along x is "more diverse".
        ax.annotate("", xy=(0.80, 1.0), xytext=(0.20, 1.0),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#bcbcbc",
                                    linewidth=0.45 * S, shrinkA=0, shrinkB=0,
                                    relpos=(0, 0)),
                    annotation_clip=False)
        for xf, txt, ha in ((0.19, "less diverse", "right"),
                            (0.81, "more diverse", "left")):
            ax.annotate(txt, xy=(xf, 1.0), xycoords="axes fraction",
                        xytext=(0, ROW_ARROW * S), textcoords="offset points",
                        ha=ha, va="center",
                        fontsize=st.pt(MEANLAB_PT), color="#9a9a9a")

        # Identical two-line statistics card in both panels (the strings have
        # the same character counts, so the boxes come out the same width).
        ax.set_ylim(0.0, ax.get_ylim()[1] * 1.22)
        ax.text(
            0.985, 0.988,
            f"AUC = {stats['auc']:.2f}\nCohen's d = {stats['cohens_d']:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=st.pt(ANNOT_PT), color=st.INK_MUTED, linespacing=1.45,
            bbox=dict(facecolor="white", edgecolor="#dcdcdc", alpha=0.94,
                      boxstyle="round,pad=0.4", linewidth=0.5 * S),
            zorder=7,
        )

        ax.set_xlabel(PANEL_XLABEL.get(title, "Diversity score"), labelpad=1.5 * S)
        if ax is flat_axes[0]:
            ax.set_ylabel("Density", labelpad=2.0 * S)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 4, 5, 10]))
        st.tidy_axes(ax)
        for gl in ax.get_ygridlines():
            gl.set_linewidth(0.35 * S)
            gl.set_color("#e3e3e3")
        ax.spines["left"].set_bounds(*ax.get_ylim())
        ax.spines["bottom"].set_visible(False)
        ax.axhline(ax.get_ylim()[0], color=st.AXIS_GRAY, linewidth=0.7 * S,
                   zorder=4, clip_on=False)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    # Flat legend row above the panels; no suptitle (the caption carries it,
    # including the group size K=0).
    # Constrained layout equalises *columns*, decorations included, so the panel
    # carrying the y-axis label comes out narrower. Freeze the layout and place
    # the axes by hand: identical widths and heights, one small gutter, and a
    # reserved band at the top for the legend.
    if len(flat_axes) == 2:
        fig.canvas.draw()
        p0, p1 = (a.get_position() for a in flat_axes)
        left, right, gutter, legend_band = p0.x0, p1.x1, 0.055, 0.085
        width = (right - left - gutter) / 2.0
        height = p0.height - legend_band
        fig.set_layout_engine("none")
        flat_axes[0].set_position([left, p0.y0, width, height])
        flat_axes[1].set_position([left + width + gutter, p1.y0, width, height])

    fig.legend(
        *handles_labels, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, 0.998), frameon=False,
        borderpad=0.0, handlelength=1.2, handletextpad=0.45,
        columnspacing=1.6, labelcolor=st.INK,
    )
    out = args.outdir / "fig1_score_validity_combined.png"
    st.save(fig, out.with_suffix(".pdf"), out)
    print(f"[write] {out}", flush=True)

    csv_path = args.outdir / "score_validity_combined_summary.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        cols = list(next(iter(summary.values())).keys())
        writer.writerow(["metric", "verdict"] + cols)
        for name, s in summary.items():
            writer.writerow([name, verdict(s)] + [s[c] for c in cols])
    print(f"[write] {csv_path}", flush=True)


if __name__ == "__main__":
    main()
