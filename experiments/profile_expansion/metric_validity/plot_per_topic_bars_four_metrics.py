#!/usr/bin/env python3
"""Per-topic validity bars for all four diversity metrics (redraw of fig2).

For each metric and each agenda topic, compare the metric's per-topic diversity
on same-profile groups vs different-profile groups of K runs. A valid metric is
LOW for same-profile and HIGH for different-profile at (nearly) every topic.

Four panels, one per metric, so the three semantic-family metrics show their own
per-topic aggregation:
    Semantic    avg of ALL pairwise cosine distances per topic
    Group       avg k-nearest cosine distance per topic
    Simulation  avg nearest-neighbor cosine distance per topic
    Behavior    per-attribute Jaccard over extracted VALUE strings per topic

Styled after results/fig/plot_topic_model_bars_clean.py (same palette / short
stage labels). Reuses the embedding + behavior extraction caches; no LLM calls.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
import matplotlib.pyplot as plt

from experiments.profile_expansion.figures import acl_fig_style as st  # noqa: E402  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402

# Palette from results/fig/plot_topic_model_bars_clean.py
MODEL_COLORS = {
    "eeyore": "#ffbe78", "angel": "#1f77b4",
    "patient_psi": "#98df8a", "roleplay_doh": "#f28e8c",
}
# Slightly desaturated blue / light red, per the figure spec.
WITHIN_COLOR = "#4C8EDA"    # same-profile group
BETWEEN_COLOR = "#F28E8E"   # different-profile group

# Type ramp copied from results/fig/behavior_d/topic_behavior_d.pdf:
# 15.75 / 16.8 / 16.1 nominal pt on the SCALE=2 canvas.
# Type hierarchy: panel title > axis title > legend > tick labels > annotation.
# Nominal pt on the SCALE=2 canvas. Ticks are deliberately the smallest text:
# the label tilt is bounded by tick height (see fit_rotation), so 17pt ticks
# would force a near-vertical tilt.
TITLE_PT = 17.0 / 2.0
SUBTITLE_PT = 12.5 / 2.0
LABEL_PT = 15.5 / 2.0
LEGEND_PT = 16.0 / 2.0
TICK_PT = 13.5 / 2.0
ANNOT_PT = 12.5 / 2.0
PANEL_HEIGHT = 3.15  # printed inches for the single row of panels
# Preferred tilt. Two rotated labels collide unless the perpendicular gap
# between their baselines -- slot_width * sin(theta) -- exceeds a line height,
# so the true minimum depends on the panel width and tick size. fit_rotation()
# measures the rendered axes and raises the angle only if this floor is too
# shallow; at one textwidth with two panels of 14 stages it lands near 60 deg.
X_TICK_ROTATION = 30
# Panel heading + grey subtitle, keyed by the internal metric title. The left
# panel is named "Semantic Diversity" to match the wording used elsewhere in the
# paper; the subtitle states the measure actually plotted.
# Shorter than the shared SHORT_LABEL set: a tilted label overhangs its tick
# sideways by len*cos(theta), and here that overhang is what pushes the two
# panels apart. The "Risk/" prefix is dropped -- the caption names the group.
PANEL_LABEL_OVERRIDE = {
    "presenting_problem": "Present",
    "risk_suicide_self_harm": "SI",
    "risk_harm_others": "Harm",
    "risk_substance_use": "Subst",
}

PANEL_TITLES = {
    "Semantic Diversity (all-pairs)": ("Semantic Diversity", "all-pairs cosine similarity"),
    "Group Diversity (k-nearest)": ("Group Diversity", "k-nearest similarity"),
    "Simulation Diversity (nearest-neighbor)": ("Semantic Diversity", "nearest-neighbor similarity"),
    "Behavior Diversity (value Jaccard)": ("Behavior Diversity",
                                            "Value Jaccard similarity, logit axis"),
}

GROUP_WIDTH = 0.36   # narrower bars -> more air between topic groups


def fit_rotation(fig, ax, n_slots, font_pt, floor_deg):
    """Smallest tilt (>= floor) at which adjacent x labels clear each other."""
    import math

    fig.canvas.draw()
    box = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    slot_px = box.width / max(1e-9, (x1 - x0))
    # 1.55 em, not 1.0: DejaVu ascenders and descenders (Sym/Emo -> Sym/Beh)
    # need clearance beyond the nominal line height before they read as separate.
    line_px = 1.55 * font_pt * fig.dpi / 72.0
    if line_px >= slot_px:
        return 90.0
    return max(float(floor_deg), math.degrees(math.asin(line_px / slot_px)))


def separation_stats(within, between):
    """AUC (P[between > within]) and Cohen's d over the pooled per-stage scores."""
    from scipy.stats import mannwhitneyu

    w = np.asarray(within, dtype=float)
    b = np.asarray(between, dtype=float)
    nw, nb = w.size, b.size
    pooled_var = ((nw - 1) * w.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(1, nw + nb - 2)
    sd = float(np.sqrt(pooled_var)) if pooled_var > 0 else 0.0
    auc = float(mannwhitneyu(b, w, alternative="greater").statistic) / (nw * nb)
    return auc, ((float(b.mean()) - float(w.mean())) / sd if sd > 0 else 0.0)


BAR_ALPHA = 0.85  # slightly desaturated in print


def ERROR_KW(S):
    """Thin, recessive error bars: present, never dominant."""
    return {"elinewidth": 0.4 * S, "capthick": 0.4 * S,
            "ecolor": "#6a6a6a", "alpha": 0.9}

# Canonical agenda order + short labels (matching the reference figure).
TOPIC_ORDER = [
    "presenting_problem", "symptoms_emotions", "symptoms_behaviors", "symptoms_cognitions",
    "onset_timeline", "triggers", "impact_functioning", "current_coping", "social_support",
    "past_treatment", "risk_suicide_self_harm", "risk_harm_others", "risk_substance_use",
    "treatment_goal",
]
SHORT_LABEL = {
    "presenting_problem": "Presenting", "symptoms_emotions": "Sym/Emo",
    "symptoms_behaviors": "Sym/Beh", "symptoms_cognitions": "Sym/Cog",
    "onset_timeline": "Onset", "triggers": "Trigger", "impact_functioning": "Impact",
    "current_coping": "Coping", "social_support": "Support", "past_treatment": "PastTx",
    "risk_suicide_self_harm": "Risk/SI", "risk_harm_others": "Risk/Harm",
    "risk_substance_use": "Risk/Subst", "treatment_goal": "Goal",
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


def semantic_per_topic(score_fn: Callable, group: List[Dict[str, Any]],
                       cache: Dict[str, np.ndarray]) -> Dict[str, float]:
    with contextlib.redirect_stdout(io.StringIO()):
        result = score_fn(group, embedding_cache=cache)
    profs = result.get("profiles") or []
    if not profs:
        return {}
    out: Dict[str, float] = {}
    for t in profs[0].get("topics", []):
        v = t.get("avg_pairwise_cosine_distance")
        if v is not None:
            out[str(t.get("topic_key"))] = float(v)
    return out


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


def accumulate(per_topic_fn, groups) -> Dict[str, List[float]]:
    acc: Dict[str, List[float]] = defaultdict(list)
    for g in groups:
        for tk, v in per_topic_fn(g).items():
            acc[tk].append(v)
    return acc


def ci95(values: List[float]) -> Tuple[float, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return 0.0, 0.0
    m = float(a.mean())
    h = 1.96 * float(a.std(ddof=1)) / np.sqrt(a.size) if a.size > 1 else 0.0
    return m, h


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path,
                    default=layout.RESULTS_DIR / "patient_psi_agenda_runs25.jsonl")
    ap.add_argument("--record-cache", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=layout.VALIDITY_DIR)
    ap.add_argument("--embed-cache", type=Path, default=layout.VALIDITY_DIR / "embed_cache.npz")
    ap.add_argument("--group-size", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=200)
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
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    K = args.group_size

    records = load_records(args.input)
    by_profile: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_profile[rec.get("profile_id")].append(rec)
    profiles = [p for p, runs in by_profile.items() if len(runs) >= K]
    print(f"[load] {len(records)} records; {len(profiles)} profiles >= {K} runs", flush=True)

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

    within_groups = [rng.sample(by_profile[rng.choice(profiles)], K) for _ in range(args.n_samples)]
    between_groups = [[rng.choice(by_profile[pid]) for pid in rng.sample(profiles, K)]
                      for _ in range(args.n_samples)]

    all_metrics = {
        "semantic": ("Semantic Diversity (all-pairs)",
         lambda g: semantic_per_topic(score_semantic_diversity, g, cache),
         lambda g, i: semantic_per_topic(score_semantic_diversity, relabel(g, f"MIX_{i}"), cache)),
        "group": ("Group Diversity (k-nearest)",
         lambda g: semantic_per_topic(score_group_diversity, g, cache),
         lambda g, i: semantic_per_topic(score_group_diversity, relabel(g, f"MIX_{i}"), cache)),
        "simulation": ("Simulation Diversity (nearest-neighbor)",
         lambda g: semantic_per_topic(score_min_distance_diversity, g, cache),
         lambda g, i: semantic_per_topic(score_min_distance_diversity, relabel(g, f"MIX_{i}"), cache)),
        "behavior": ("Behavior Diversity (value Jaccard)",
         lambda g: behavior_per_topic(g, key_to_entry),
         lambda g, i: behavior_per_topic(g, key_to_entry)),
    }
    metrics = [all_metrics[name] for name in args.metrics]

    n_panels = len(metrics)
    nrows, ncols = 1, n_panels
    st.apply_style(tick=TICK_PT, label=LABEL_PT, legend=LEGEND_PT, title=TITLE_PT)
    S = st.SCALE
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=st.size(st.width_for(args.width), PANEL_HEIGHT),
        layout="constrained",
    )
    # Tight gutter: every inch not spent on the gap widens both panels, which
    # is what gives 14 categories room to breathe.
    fig.get_layout_engine().set(w_pad=0.015 * S, h_pad=0.04 * S, wspace=0.025)
    flat_axes = np.atleast_1d(axes).ravel()
    handles_labels = None
    rot_axes = []
    for panel_idx, (ax, (title, within_fn, between_fn)) in enumerate(zip(flat_axes, metrics)):
        print(f"[score] {title} ...", flush=True)
        w_acc = accumulate(within_fn, within_groups)
        b_acc = accumulate(lambda g, _i=[0]: between_fn(g, _bump(_i)), between_groups)

        topics = [t for t in TOPIC_ORDER if t in w_acc or t in b_acc]
        x = np.arange(len(topics))
        wm = np.array([ci95(w_acc.get(t, []))[0] for t in topics], dtype=float)
        wh = np.array([ci95(w_acc.get(t, []))[1] for t in topics], dtype=float)
        bm = np.array([ci95(b_acc.get(t, []))[0] for t in topics], dtype=float)
        bh = np.array([ci95(b_acc.get(t, []))[1] for t in topics], dtype=float)

        # Behavior diversity hugs 1.0; use a logit y-axis to expand the near-ceiling
        # differences (matching results/fig/behavior_d). Bars are drawn from a
        # positive baseline and capped just below 1 so the logit axis stays finite.
        logit = "Behavior" in title
        if logit:
            bounds = np.concatenate([wm - wh, wm + wh, bm - bh, bm + bh])
            bounds = bounds[np.isfinite(bounds)]
            y_min = float(bounds.min()) if bounds.size else 0.0
            y_max = float(bounds.max()) if bounds.size else 1.0
            top_cap = 1.0 - 1e-4
            view_bottom = max(1e-3, y_min - (y_max - y_min) * 0.12)
            view_top = 1.0 - 5e-5
            for offset, means, half, color, label in (
                (-GROUP_WIDTH / 2, wm, wh, WITHIN_COLOR, "Same profile"),
                (GROUP_WIDTH / 2, bm, bh, BETWEEN_COLOR, "Different profile"),
            ):
                tops = np.clip(means, view_bottom * 1.0001, top_cap)
                hi = np.minimum(half, top_cap - tops)
                lo = np.minimum(half, tops - view_bottom)
                ax.bar(x + offset, tops - view_bottom, GROUP_WIDTH, bottom=view_bottom,
                       yerr=np.array([lo, hi]), capsize=1.0 * S, color=color,
                       edgecolor="white", linewidth=0.4 * S, label=label,
                       alpha=BAR_ALPHA, error_kw=ERROR_KW(S), zorder=3)
            ax.set_yscale("logit")
            ax.set_ylim(view_bottom, view_top)
            # Fewer ticks: a full logit ladder (0.7 0.8 0.9 0.95 0.98 0.99 ...)
            # reads as unevenly spaced clutter. Evenly spaced *values* are not
            # possible on a logit axis, so show a short, decisive set instead.
            candidate = [0.7, 0.9, 0.99, 0.999]
            ticks = st.thin_ticks(
                fig, ax,
                [t for t in candidate if view_bottom < t < view_top],
                st.pt(TICK_PT),
            )
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{t:g}" for t in ticks])
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            if ax is flat_axes[0]:
                ax.set_ylabel("Diversity score", labelpad=2.0 * S)
            st.draw_axis_break(ax)
        else:
            ax.bar(x - GROUP_WIDTH / 2, wm, GROUP_WIDTH, yerr=wh, capsize=1.0 * S,
                   color=WITHIN_COLOR, edgecolor="white", linewidth=0.4 * S,
                   label="Same profile", alpha=BAR_ALPHA,
                   error_kw=ERROR_KW(S), zorder=3)
            ax.bar(x + GROUP_WIDTH / 2, bm, GROUP_WIDTH, yerr=bh, capsize=1.0 * S,
                   color=BETWEEN_COLOR, edgecolor="white", linewidth=0.4 * S,
                   label="Different profile", alpha=BAR_ALPHA,
                   error_kw=ERROR_KW(S), zorder=3)
            ax.set_ylabel("Diversity score", labelpad=2.0 * S)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [PANEL_LABEL_OVERRIDE.get(t, SHORT_LABEL.get(t, t)) for t in topics],
            ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.62, len(topics) - 0.38)
        rot_axes.append((ax, len(topics)))
        # Panel label + title, with a smaller grey subtitle naming the measure.
        tag = f"({chr(ord('a') + panel_idx)}) "
        head, sub = PANEL_TITLES.get(title, (title, ""))
        ax.set_title(tag + head, fontsize=st.pt(TITLE_PT), fontweight="bold",
                     pad=(2.0 + 1.35 * SUBTITLE_PT) * S)
        if sub:
            ax.text(0.5, 1.012, f"({sub})", transform=ax.transAxes,
                    ha="center", va="bottom",
                    fontsize=st.pt(SUBTITLE_PT), color=st.INK_MUTED)

        # Separation statistics, pooled over this panel's stages, tucked into
        # the upper-right corner.
        w_all = [v for vals in w_acc.values() for v in vals]
        b_all = [v for vals in b_acc.values() for v in vals]
        if len(w_all) > 1 and len(b_all) > 1:
            auc, d = separation_stats(w_all, b_all)
            ax.text(0.99, 0.99, f"AUC = {auc:.2f}\nd = {d:.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=st.pt(ANNOT_PT), color=st.INK_MUTED,
                    linespacing=1.3, zorder=6)

        st.tidy_axes(ax)
        for gl in ax.get_ygridlines():          # thinner + lighter than default
            gl.set_linewidth(0.35 * S)
            gl.set_color("#e3e3e3")
        if not logit:
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.spines["left"].set_bounds(*ax.get_ylim())
        ax.spines["bottom"].set_visible(False)
        ax.axhline(ax.get_ylim()[0], color=st.AXIS_GRAY, linewidth=0.7 * S,
                   zorder=4, clip_on=False)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()

    for ax in flat_axes[n_panels:]:
        ax.set_visible(False)

    # Tilt the stage labels by the smallest angle that keeps them apart.
    rotation = max(
        fit_rotation(fig, ax_, n_, st.pt(TICK_PT), X_TICK_ROTATION)
        for ax_, n_ in rot_axes
    )
    for ax_, _ in rot_axes:
        for lbl in ax_.get_xticklabels():
            lbl.set_rotation(rotation)
    print(f"[layout] stage labels tilted {rotation:.0f} deg", flush=True)

    # Constrained layout gives each *column* equal width, decorations included,
    # so the panel carrying the y-axis label ends up narrower than its neighbour
    # and the leftover space piles up in the middle. Freeze the layout and lay
    # the axes out by hand: identical widths, one small fixed gutter.
    if len(flat_axes) == 2:
        fig.canvas.draw()
        p0, p1 = (a.get_position() for a in flat_axes)
        left, right, gutter = p0.x0, p1.x1, 0.055
        width = (right - left - gutter) / 2.0
        legend_band = 0.085  # freed at the top for the legend row
        height = p0.height - legend_band
        fig.set_layout_engine("none")
        flat_axes[0].set_position([left, p0.y0, width, height])
        flat_axes[1].set_position([left + width + gutter, p1.y0, width, height])

    # Flat legend row above the panels; no suptitle.
    # Figure-level so it centres on the page, not on one panel; the layout is
    # frozen by now, so it needs an explicit anchor inside the canvas.
    fig.legend(
        *handles_labels, ncol=2, loc="upper center",
        bbox_to_anchor=(0.5, 0.998), frameon=False,
        borderpad=0.0, handlelength=1.2, handletextpad=0.45,
        columnspacing=1.6, labelcolor=st.INK,
    )
    out = args.outdir / "fig2_per_topic_bars_four_metrics.png"
    st.save(fig, out.with_suffix(".pdf"), out)
    print(f"[write] {out}", flush=True)


def _bump(counter: List[int]) -> int:
    counter[0] += 1
    return counter[0] - 1


if __name__ == "__main__":
    main()
