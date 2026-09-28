#!/usr/bin/env python3
"""The two paper figures for the profile-expansion experiment.

This replaces the four separate figures (summary / simulation-by-stage /
behavior-by-stage / alignment-by-aspect) with two multi-panel figures, so the
paper carries one legend and one caption per level of detail instead of four:

  Figure 1 -- Overall model performance          <figures>/overall_model_performance.pdf
      (a) Simulation Diversity  (b) Behavior Diversity  (c) Profile Alignment
      1 x 3 bar panels, one bar per model, with 95% CI whiskers.

  Figure 2 -- Stage- and aspect-level performance  <figures>/stage_aspect_performance.pdf
      (a) Simulation Diversity by Stage
      (b) Behavior Diversity by Stage
      (c) Profile Alignment by Aspect
      Three stacked panels of vertical grouped bars, four bars per stage/aspect,
      with 95% CI whiskers. No value labels: 56 of them would not fit.

  Figure 3 -- Diversity vs. runs per profile      <figures>/diversity_by_runs.pdf
      (a) Simulation Diversity  (b) Behavior Diversity
      Two line panels with 95% bootstrap CI bands, replacing the two separate
      run_number/*_runs_line.pdf figures.

  Figure 4 -- Diversity vs. fixed attributes      <figures>/diversity_by_fixed_attributes.pdf
      (a) Simulation Diversity  (b) Behavior Diversity
      Plots the measured run-4 fixed-attribute metrics, matching the per-metric
      figures of fixed_attributes/plot_fixed_attr_diversity_four_metrics.py.

All three figures use linear axes only. Panels whose axis does not start at zero carry
a break mark on the y spine (see ``DRAW_AXIS_BREAKS``), because a bar's length is
read as its magnitude.

"Simulation Diversity" is the name used throughout; the underlying metric key is
still ``min_distance_diversity``.

Sizing convention
-----------------
Unlike ``acl_fig_style`` (which authors at 2x the printed size), these figures
are authored at their *printed* size: 7.2in wide, to be included with

    \\includegraphics[width=\\textwidth]{...}   % inside figure*

The nominal font sizes below are therefore the printed sizes -- but only if the
target ``\\textwidth`` is ~7.2in. ACL's ``\\textwidth`` is 6.3in, so LaTeX
shrinks the PDF by ~0.89 and 8pt lands at ~7pt. Set ``FONT_SCALE = 1.14``
(= 7.2 / 6.3) to compensate, or ``FIG_WIDTH = 6.3`` to author at the ACL width
directly.

Both figures are saved with ``bbox_inches="tight"`` so the shared legend, which
is anchored just above the canvas, is not cropped; the saved page is therefore a
touch smaller than the nominal size and LaTeX scales it by slightly less.

Numbers are identical to the figures this replaces: per-profile means with 95%
bootstrap CIs (10k resamples, seed 42); per-stage scores are adjusted by each
profile's ``informative_run_ratio``; aspect scores are rescaled from 1-5 to 0-1.

Usage (from the repository root):
    python -m experiments.profile_expansion.figures.plot_paper_figures
"""

from __future__ import annotations

import argparse
import json
import math
import re
from glob import glob
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Model order and display names live in one place, shared with the older figures.
from experiments.profile_expansion.figures.acl_fig_style import MODEL_DISPLAY, MODEL_ORDER


# --- Type ramp (printed pt; see the sizing note in the module docstring) ------
# 1.56 scales the whole ramp up by 56% off the profile_exp_summary.pdf match
# recorded below (1.30, then a further 20%); 1.0 restores that match exactly.
FONT_SCALE = 1.56

# This ramp reproduces the printed type size of the older
# summary/profile_exp_summary.pdf. That figure is authored on an 18.16in-wide
# canvas with a 22.5pt tick size, so once LaTeX scales it to a two-column width
# its ticks print at 22.5 * (W / 18.16). Matching that here means multiplying its
# nominal sizes by our_width / 18.16 ~= 0.385.
FS_BASE = 8.7
FS_TITLE = 10.1
FS_LABEL = 9.3
FS_XTICK = 8.7
FS_YTICK = 8.7
FS_LEGEND = 8.9
# Figure 1 fits four value labels across a ~2.1in panel, so this one element sits
# well below the rest of the ramp; at 3 decimals it is what limits the figure.
FS_VALUE = 6.9

INK = "#1a1a1a"
AXIS_GRAY = "0.55"
GRID_GRAY = "0.84"
SEPARATOR_GRAY = "0.88"
BREAK_GRAY = "0.45"
ERROR_GRAY = "0.30"

# --- Model identity ----------------------------------------------------------
# A darkened variant of the project palette: the original pale orange/green sat
# below 3:1 contrast on white. Identity still never rests on colour alone --
# Angel (the proposed system) carries a hatch, and every bar sits above a named
# x tick.
MODEL_COLORS: Dict[str, str] = {
    "eeyore": "#FDB366",
    "angel": "#2C7FB8",
    "patient_psi": "#8BD17C",
    "roleplay_doh": "#F28E8E",
}
MODEL_HATCHES: Dict[str, str] = {"angel": "////"}
# Figure 3 draws lines, so identity there rests on colour + marker shape, and
# Angel additionally on a heavier stroke.
MODEL_MARKERS: Dict[str, str] = {
    "eeyore": "o",
    "angel": "D",
    "patient_psi": "s",
    "roleplay_doh": "^",
}
HIGHLIGHT_MODEL = "angel"
BAR_EDGE_F1 = "0.35"
BAR_EDGE_F2 = "0.45"  # thinner bars, so a lighter outline

# --- Figure 1: metrics -------------------------------------------------------
# (panel title, metric key, per-profile value field, y range, y ticks, tick
# decimals). Each metric occupies a different part of the 0-1 range, so the
# panels are not forced onto a shared axis.
OVERALL_SPECS = [
    ("Simulation Diversity", "min_distance_diversity", "min_distance_diversity",
     (0.05, 0.45), [0.05, 0.15, 0.25, 0.35, 0.45], 2),
    ("Behavior Diversity", "behavior_diversity", "behavior_diversity",
     (0.90, 1.00), [0.90, 0.925, 0.95, 0.975, 1.00], 3),
    ("Profile Alignment", "profile_alignment", "score",
     (0.80, 1.00), [0.80, 0.85, 0.90, 0.95, 1.00], 2),
]

# Bars read their length as a magnitude, so any panel zoomed above zero gets a
# break mark on the y spine. Set False to drop them.
DRAW_AXIS_BREAKS = True

# --- Figure 2: groups --------------------------------------------------------
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
# Symp. = symptoms, Emo. = emotion, Beh. = behavior, Cog. = cognition,
# Tx = treatment. En dash, not a slash, inside the compound labels.
TOPIC_LABELS = {
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
# Light rules after Impact / Past Tx / Substance, splitting the stages into
# clinical presentation | resources & history | risk assessment | goals.
TOPIC_GROUP_BREAKS_AFTER = ("impact_functioning", "past_treatment", "risk_substance_use")

ASPECT_ORDER = [
    "presenting_problem_alignment",
    "symptom_alignment",
    "emotion_alignment",
    "background_alignment",
    "personality_interpersonal_alignment",
]
ASPECT_LABELS = {
    "presenting_problem_alignment": "Presenting",
    "symptom_alignment": "Symptoms",
    "emotion_alignment": "Emotion",
    "background_alignment": "Background",
    "personality_interpersonal_alignment": "Personality",
}
ASPECT_SCALE = 5.0  # aspect_scores_1_to_5 -> 0-1, to match the overall score

# (y range, y ticks) per Figure 2 panel. The floor is lowered automatically if a
# CI whisker would otherwise be cut off -- see ``_fit_floor``.
STAGE_SIM_YLIM = (0.0, 0.55)
STAGE_SIM_YTICKS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
STAGE_BEH_YLIM = (0.60, 1.005)
STAGE_BEH_YTICKS = [0.60, 0.70, 0.80, 0.90, 1.00]  # 0.95 dropped: too tight now
ASPECT_YLIM = (0.79, 1.005)
ASPECT_YTICKS = [0.80, 0.85, 0.90, 0.95, 1.00]

# --- Figure 3: runs ----------------------------------------------------------
# (panel title, metric key, per-profile value field, y-axis label)
RUNS_SPECS = [
    ("Simulation Diversity", "min_distance_diversity", "min_distance_diversity",
     "Simulation Diversity"),
    ("Behavior Diversity", "behavior_diversity", "behavior_diversity",
     "Behavior Diversity"),
]
# Verified against the inputs: every profile in ``*.metrics.run{N}.json`` carries
# num_runs == N, i.e. N runs pooled per profile. Hence "Runs per Profile" rather
# than "Run number", which would read as the index of a single run.
RUNS_XLABEL = "Runs per Profile"
RUNS_FILENAME_RE = re.compile(r"(.+)_agenda_runs25\.metrics\.run(\d+)\.json$")
RUNS_BAND_ALPHA = 0.13
RUNS_LINEWIDTH = {"angel": 2.2}
RUNS_LINEWIDTH_DEFAULT = 1.5
RUNS_MARKEVERY = 3  # 23 runs per line; every marker would be a smear

# --- Figure 4: fixed attributes ----------------------------------------------
# (panel title, section key, per-profile field, skip unscored, y label, y range,
# y ticks)
FIXATTR_SPECS = [
    ("Simulation Diversity", "min_distance_diversity", "min_distance_diversity",
     False, "Simulation Diversity", (0.0, 0.48), [0.0, 0.1, 0.2, 0.3, 0.4]),
    ("Behavior Diversity", "behavior_diversity", "behavior_diversity",
     True, "Behavior Diversity", (0.84, 1.005), [0.85, 0.90, 0.95, 1.00]),
]
FIXATTR_XLABEL = "Number of Fixed Attributes"
# profile_id is <base_profile>_k<K>_s<S>; K is the number of fixed attributes.
FIXATTR_VARIANT_RE = re.compile(r"^(.+)_k(\d+)_s(\d+)$")
FIXATTR_MODEL_RE = re.compile(r"fixattr_variant_(.+)\.metrics\.run\d+\.json$")
FIXATTR_K_RANGE = (1, 22)
# 5000 resamples is what fix_attr/fig/*_run4.csv was written with; keeping it means
# this figure can be diffed against those files.
FIXATTR_N_BOOTSTRAP = 5000
FIXATTR_BAND_ALPHA = 0.12
FIXATTR_LINEWIDTH = {"angel": 2.0}
FIXATTR_LINEWIDTH_DEFAULT = 1.4
FIXATTR_MARKERSIZE = {"angel": 4.6}
FIXATTR_MARKERSIZE_DEFAULT = 3.8

# --- Canvas ------------------------------------------------------------------
FIG_WIDTH = 7.2
FIG1_HEIGHT = 3.0
FIG2_HEIGHT = 7.3
FIG3_HEIGHT = 2.6
FIG4_HEIGHT = 2.9
FIG2_HEIGHT_RATIOS = [1.2, 1.2, 0.70]
# hspace is a fraction of the mean panel height, but the gap it has to clear -- a
# band of rotated tick labels plus the next panel's title -- is a fixed ~0.9in at
# this type size. So it has to grow as the canvas shrinks, and again whenever the
# ticks grow: 0.62 at a 7.3in height here, 0.48 at 5.8in with a 30%-smaller ramp.
# Steeper rotation does NOT help: it makes the label band taller, not shorter.
FIG2_HSPACE = 0.72
# Panel (c) has 5 groups against 14 in panels (a) and (b), so it is centred at a
# fraction of the width instead of stretching five groups across the full span.
# The outer two entries are empty margins; widen the middle one to ~0.76 for a
# larger panel.
PANEL_C_WIDTH_RATIOS = [0.18, 0.64, 0.18]

# The shallowest tilt that keeps adjacent labels a full line-height apart, which
# is also the one that costs the least vertical space (band height grows with the
# angle). At this ramp the 14 stage labels need >= 24 degrees over their 0.46in
# pitch; panel (c)'s five wide groups need only ~15.
STAGE_TICK_ROTATION = 26
ASPECT_TICK_ROTATION = 18
# Figure 1: four model names over a ~0.5in pitch; the limit at this ramp is ~21.
MODEL_TICK_ROTATION = 30


# ============================== style ========================================
def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": FS_BASE * FONT_SCALE,
            "axes.titlesize": FS_TITLE * FONT_SCALE,
            "axes.labelsize": FS_LABEL * FONT_SCALE,
            "xtick.labelsize": FS_XTICK * FONT_SCALE,
            "ytick.labelsize": FS_YTICK * FONT_SCALE,
            "legend.fontsize": FS_LEGEND * FONT_SCALE,
            "axes.linewidth": 0.7,
            "axes.edgecolor": AXIS_GRAY,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": AXIS_GRAY,
            "ytick.color": AXIS_GRAY,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 0.0,
            "ytick.major.size": 2.5,
            "hatch.linewidth": 0.6,  # thin bars need a fine hatch
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,  # TrueType, not Type-3 (camera-ready safe)
            "ps.fonttype": 42,
        }
    )


def tidy_panel(ax, grid_width: float = 0.6) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_GRAY)
    ax.grid(axis="y", linestyle=":", linewidth=grid_width, color=GRID_GRAY, zorder=0)
    ax.set_axisbelow(True)


def draw_axis_break(ax, size_frac: float = 0.013) -> None:
    """Two slashes at the foot of the y spine: this axis does not start at zero."""
    kwargs = dict(
        transform=ax.transAxes,
        color=BREAK_GRAY,
        clip_on=False,
        linewidth=0.8,
        solid_capstyle="round",
        zorder=10,
    )
    for offset in (0.0, 0.025):
        ax.plot(
            [-size_frac * 0.7, size_frac * 0.7],
            [offset - size_frac, offset + size_frac],
            **kwargs,
        )


def bar_legend_handles(models: Sequence[str], bar_edge: str) -> List[Patch]:
    return [
        Patch(
            facecolor=MODEL_COLORS.get(m, "#767676"),
            edgecolor="white" if m == HIGHLIGHT_MODEL else bar_edge,
            linewidth=0.7,
            hatch=MODEL_HATCHES.get(m),
            label=MODEL_DISPLAY.get(m, m),
        )
        for m in models
    ]


def line_legend_handles(models: Sequence[str]) -> List[Line2D]:
    """Figure 3: the legend keys are lines with their marker."""
    return [
        Line2D(
            [],
            [],
            color=MODEL_COLORS.get(m, "#767676"),
            marker=MODEL_MARKERS.get(m, "o"),
            markersize=4.0 if m == HIGHLIGHT_MODEL else 3.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=RUNS_LINEWIDTH.get(m, RUNS_LINEWIDTH_DEFAULT),
            label=MODEL_DISPLAY.get(m, m),
        )
        for m in models
    ]


def add_shared_legend(fig, handles: Sequence[Any], y: float) -> None:
    fig.legend(
        handles=list(handles),
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=len(handles),
        frameon=False,
        handlelength=1.5,
        handletextpad=0.5,
        columnspacing=1.5,
        borderpad=0.0,
        labelcolor=INK,
        fontsize=FS_LEGEND * FONT_SCALE,
    )


# ============================== data =========================================
def as_finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_docs(clean_dir: Path, run_number: int) -> Dict[str, dict]:
    docs: Dict[str, dict] = {}
    for model in MODEL_ORDER:
        path = clean_dir / f"{model}_agenda_runs5.metrics.combined.run{run_number}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")
        docs[model] = json.loads(path.read_text())
    return docs


def bootstrap_mean_ci(
    values: Sequence[float], rng: np.random.Generator, n_bootstrap: int
) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean)
    indices = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    sample_means = arr[indices].mean(axis=1)
    low, high = np.percentile(sample_means, [2.5, 97.5])
    return (mean, float(low), float(high))


def profile_weight(profile: Dict[str, Any]) -> float:
    """A profile's informative_run_ratio, recomputed from counts if absent."""
    ratio = as_finite_float(profile.get("informative_run_ratio"))
    if ratio is None:
        informative = as_finite_float(profile.get("informative_topic_runs"))
        total = as_finite_float(profile.get("total_topic_runs"))
        if informative is not None and total and total > 0:
            ratio = informative / total
    return 1.0 if ratio is None else max(0.0, ratio)


def overall_values(doc: dict, metric_key: str, value_field: str) -> List[float]:
    """Per-profile scores behind one Figure 1 panel."""
    if metric_key == "profile_alignment":
        rows = doc["profile_alignment"]["by_model_profile"]
        return [
            float(row[value_field]) for row in rows if row.get(value_field) is not None
        ]
    profiles = doc[metric_key]["profiles"]
    if metric_key == "behavior_diversity":
        profiles = [p for p in profiles if p.get("scored", True)]
    return [
        float(p[value_field])
        for p in profiles
        if as_finite_float(p.get(value_field)) is not None
    ]


def topic_values(
    doc: dict, metric_key: str, value_field: str, adjust_by_ratio: bool
) -> Dict[str, List[float]]:
    """topic_key -> per-profile scores, optionally ratio-adjusted."""
    out: Dict[str, List[float]] = {}
    for profile in doc[metric_key]["profiles"]:
        if metric_key == "behavior_diversity" and not profile.get("scored", True):
            continue
        weight = profile_weight(profile) if adjust_by_ratio else 1.0
        for topic in profile.get("topics", []):
            topic_key = topic.get("topic_key")
            score = as_finite_float(topic.get(value_field))
            if not isinstance(topic_key, str) or score is None:
                continue
            out.setdefault(topic_key, []).append(score * weight)
    return out


def aspect_values(doc: dict) -> Dict[str, List[float]]:
    """aspect_key -> per-profile scores rescaled to 0-1."""
    out: Dict[str, List[float]] = {}
    for row in doc["profile_alignment"]["by_model_profile"]:
        for aspect_key, raw in (row.get("aspect_scores_1_to_5") or {}).items():
            score = as_finite_float(raw)
            if not isinstance(aspect_key, str) or score is None:
                continue
            out.setdefault(aspect_key, []).append(score / ASPECT_SCALE)
    return out


def collect_run_stats(
    input_glob: str, metric_key: str, value_field: str, n_bootstrap: int, seed: int
) -> Dict[str, Dict[int, Tuple[float, float, float]]]:
    """model -> runs-per-profile -> (mean, ci_low, ci_high).

    Path order and the single shared generator match the older
    run_number/plot_diversity_by_run_ci.py, so the bands are bit-identical to the
    two figures this replaces.
    """
    paths = sorted(Path(p) for p in glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {input_glob}")

    rng = np.random.default_rng(seed)
    stats: Dict[str, Dict[int, Tuple[float, float, float]]] = {}
    for path in paths:
        match = RUNS_FILENAME_RE.match(path.name)
        if match is None:
            continue
        model, runs = match.group(1), int(match.group(2))
        doc = json.loads(path.read_text())
        profiles = doc.get(metric_key, {}).get("profiles", [])
        values = [
            v
            for profile in profiles
            if not (metric_key == "behavior_diversity" and not profile.get("scored", True))
            for v in [as_finite_float(profile.get(value_field))]
            if v is not None
        ]
        if not values:
            continue
        stats.setdefault(model, {})[runs] = bootstrap_mean_ci(values, rng, n_bootstrap)
    return stats


def collect_fixattr_stats(
    input_glob: str, section: str, field: str, skip_unscored: bool, n_bootstrap: int, seed: int
) -> Dict[str, Dict[int, Tuple[float, float, float]]]:
    """model -> K (fixed attributes) -> (mean, ci_low, ci_high), measured values.

    Mirrors ``plot_metric`` in fixed_attributes/plot_fixed_attr_diversity_four_metrics.py
    -- one generator consumed in (model, K) order -- so the output matches that
    script's *_run4.csv exactly.
    """
    paths = sorted(Path(p) for p in glob(input_glob))
    if not paths:
        raise FileNotFoundError(f"No files matched: {input_glob}")

    grouped: Dict[Tuple[str, int], List[float]] = {}
    for path in paths:
        match = FIXATTR_MODEL_RE.match(path.name)
        model = match.group(1) if match else path.stem
        doc = json.loads(path.read_text(encoding="utf-8"))
        for profile in doc.get(section, {}).get("profiles", []) or []:
            if skip_unscored and not profile.get("scored", True):
                continue
            variant = FIXATTR_VARIANT_RE.match(str(profile.get("profile_id", "")))
            score = as_finite_float(profile.get(field))
            if variant is None or score is None:
                continue
            grouped.setdefault((model, int(variant.group(2))), []).append(score)

    k_lo, k_hi = FIXATTR_K_RANGE
    ks = sorted({k for (_m, k) in grouped if k_lo <= k <= k_hi})
    rng = np.random.default_rng(seed)
    stats: Dict[str, Dict[int, Tuple[float, float, float]]] = {}
    for model in MODEL_ORDER:
        for k in ks:
            values = grouped.get((model, k), [])
            if not values:
                continue
            stats.setdefault(model, {})[k] = bootstrap_mean_ci(values, rng, n_bootstrap)
    return stats


Stats = Dict[str, Dict[str, Tuple[float, float, float]]]  # row/metric -> model -> stat


def compute_all_stats(
    docs: Dict[str, dict], n_bootstrap: int, seed: int, adjust_by_ratio: bool
) -> Dict[str, Stats]:
    """Every number in both figures, keyed by block name."""
    rng = np.random.default_rng(seed)

    overall: Stats = {}
    for _, metric_key, value_field, *_ in OVERALL_SPECS:
        overall[metric_key] = {
            model: bootstrap_mean_ci(
                overall_values(docs[model], metric_key, value_field), rng, n_bootstrap
            )
            for model in MODEL_ORDER
        }

    stage_blocks: Dict[str, Stats] = {}
    for block, metric_key, value_field in (
        ("stage_simulation", "min_distance_diversity", "avg_nearest_neighbor_cosine_distance"),
        ("stage_behavior", "behavior_diversity", "behavior_diversity"),
    ):
        per_model = {
            model: topic_values(docs[model], metric_key, value_field, adjust_by_ratio)
            for model in MODEL_ORDER
        }
        stage_blocks[block] = {
            topic_key: {
                model: bootstrap_mean_ci(
                    per_model[model].get(topic_key, []), rng, n_bootstrap
                )
                for model in MODEL_ORDER
            }
            for topic_key in TOPIC_ORDER
        }

    per_model_aspects = {model: aspect_values(docs[model]) for model in MODEL_ORDER}
    aspects: Stats = {
        aspect_key: {
            model: bootstrap_mean_ci(
                per_model_aspects[model].get(aspect_key, []), rng, n_bootstrap
            )
            for model in MODEL_ORDER
        }
        for aspect_key in ASPECT_ORDER
    }

    return {"overall": overall, **stage_blocks, "aspect": aspects}


# ============================== Figure 1 =====================================
def _two_line_title(text: str) -> str:
    """Break a panel title at its last space.

    At the current ramp a one-line "(a) Simulation Diversity" is wider than the
    panel it sits over, so adjacent titles collide. Wrapping costs a little height
    and keeps the metric names intact, which abbreviating would not.
    """
    head, _, tail = text.rpartition(" ")
    return f"{head}\n{tail}" if head else text


def draw_overall_bar_panel(
    ax,
    stats_for_metric: Dict[str, Tuple[float, float, float]],
    title: str,
    ylim: Tuple[float, float],
    yticks: Sequence[float],
    tick_decimals: int,
    ylabel: Optional[str] = None,
    value_decimals: int = 3,
) -> None:
    x = np.arange(len(MODEL_ORDER), dtype=float)
    width = 0.62

    for i, model in enumerate(MODEL_ORDER):
        mean, ci_low, ci_high = stats_for_metric[model]
        is_focus = model == HIGHLIGHT_MODEL
        # Asymmetric whiskers: the bootstrap CI is not symmetric about the mean.
        yerr = np.array([[mean - ci_low], [ci_high - mean]], dtype=float)

        ax.bar(
            x[i],
            mean,
            width=width,
            color=MODEL_COLORS.get(model, "#767676"),
            edgecolor="white" if is_focus else BAR_EDGE_F1,
            linewidth=0.8,
            hatch=MODEL_HATCHES.get(model),
            yerr=yerr,
            error_kw={
                "elinewidth": 1.0,
                "capsize": 3,
                "capthick": 1.0,
                "ecolor": "0.20",
            },
            zorder=3,
        )
        # Offset in points, not in data units: the three panels have different
        # spans, so a data-unit offset would sit at three different distances.
        ax.annotate(
            f"{mean:.{value_decimals}f}",
            xy=(x[i], ci_high),
            xytext=(0, 2.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=FS_VALUE * FONT_SCALE,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [MODEL_DISPLAY.get(m, m) for m in MODEL_ORDER],
        rotation=MODEL_TICK_ROTATION,
        ha="right",
        rotation_mode="anchor",
    )
    for label, model in zip(ax.get_xticklabels(), MODEL_ORDER):
        if model == HIGHLIGHT_MODEL:  # the proposed system
            label.set_fontweight("bold")

    ax.set_xlim(-0.62, len(MODEL_ORDER) - 0.38)
    ax.set_ylim(*ylim)
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([f"{t:.{tick_decimals}f}" for t in yticks])
    ax.set_title(_two_line_title(title), loc="left", pad=6.0, fontweight="semibold")
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2.0)

    tidy_panel(ax, grid_width=0.7)
    ax.tick_params(axis="x", pad=1.5)
    # The view top may be raised later to fit the value labels; the spine stops at
    # the last tick either way.
    ax.spines["left"].set_bounds(*ylim)
    if DRAW_AXIS_BREAKS and ylim[0] > 0.0:
        draw_axis_break(ax)


def _fit_label_headroom(
    fig, axes: Sequence[Any], base_ylims: Sequence[Tuple[float, float]], passes: int = 3
) -> None:
    """Raise every panel's view top by the same fraction of its span.

    The value labels are anchored to a data position with a fixed offset in
    points, so how much headroom they need depends on the final panel height --
    measuring beats guessing, since a label that pokes out of the axes runs into
    the panel title. The fraction is shared across panels so all three spines
    still end at their last tick at the same height.
    """
    frac = 0.0
    for _ in range(passes):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        needed = frac
        for ax, (lo, hi) in zip(axes, base_ylims):
            if not ax.texts:
                continue
            axes_top = float(ax.transAxes.transform((0.0, 1.0))[1])
            label_top = max(
                float(t.get_window_extent(renderer=renderer).y1) for t in ax.texts
            )
            if label_top <= axes_top - 1.0:
                continue
            inv = ax.transData.inverted()
            gap = (
                inv.transform((0.0, label_top + 1.5))[1]
                - inv.transform((0.0, axes_top))[1]
            )
            needed = max(needed, (ax.get_ylim()[1] + gap - hi) / (hi - lo))
        if needed <= frac + 1e-4:
            return
        frac = needed
        for ax, (lo, hi) in zip(axes, base_ylims):
            ax.set_ylim(lo, hi + (hi - lo) * frac)


def plot_overall(stats: Stats, output_base: Path) -> None:
    apply_style()

    fig, axes = plt.subplots(
        1,
        len(OVERALL_SPECS),
        figsize=(FIG_WIDTH, FIG1_HEIGHT),
        gridspec_kw={"wspace": 0.34},
    )
    axes = np.atleast_1d(axes)

    for panel_idx, (ax, spec) in enumerate(zip(axes, OVERALL_SPECS)):
        title, metric_key, _, ylim, yticks, tick_decimals = spec
        draw_overall_bar_panel(
            ax,
            stats[metric_key],
            title=f"({'abc'[panel_idx]}) {title}",
            ylim=ylim,
            yticks=yticks,
            tick_decimals=tick_decimals,
            ylabel="Score" if panel_idx == 0 else None,
        )

    fig.subplots_adjust(top=0.78, bottom=0.26, left=0.09, right=0.99)
    _fit_label_headroom(fig, axes, [spec[3] for spec in OVERALL_SPECS])
    # y=1.06, not 1.02: the legend hangs down from its anchor, and the two-line
    # titles now reach high enough that 1.02 interleaves them.
    add_shared_legend(fig, bar_legend_handles(MODEL_ORDER, BAR_EDGE_F1), y=1.06)
    save(fig, output_base)


# ============================== Figure 2 =====================================
def _fit_floor(
    stats: Stats,
    group_keys: Sequence[str],
    ylim: Tuple[float, float],
    yticks: Sequence[float],
) -> Tuple[Tuple[float, float], List[float]]:
    """Lower the axis floor if a CI whisker would be cut off by it.

    The nominal ranges are chosen by hand, but a clipped whisker silently hides
    uncertainty, so the floor drops to the next 0.05 below the lowest CI bound
    and that value joins the tick list.
    """
    lows = [
        stats[key][model][1]
        for key in group_keys
        for model in MODEL_ORDER
        if np.isfinite(stats[key][model][1])
    ]
    if not lows or min(lows) >= ylim[0]:
        return ylim, list(yticks)
    floor = math.floor((min(lows) - 0.005) * 20.0) / 20.0
    # Drop any tick the new floor would sit almost on top of (0.55 vs 0.60).
    span = ylim[1] - floor
    kept = [t for t in yticks if t - floor > span * 0.14]
    return (floor, ylim[1]), sorted({round(floor, 2), *kept})


def draw_grouped_bars(
    ax,
    group_keys: Sequence[str],
    group_labels: Sequence[str],
    stats: Stats,
    title: str,
    ylabel: str,
    ylim: Tuple[float, float],
    yticks: Sequence[float],
    tick_decimals: int = 2,
    rotation: float = STAGE_TICK_ROTATION,
) -> None:
    x = np.arange(len(group_keys), dtype=float)
    n_models = len(MODEL_ORDER)
    total_width = 0.78
    bar_width = total_width / n_models
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width

    for i, model in enumerate(MODEL_ORDER):
        means, err_lo, err_hi = [], [], []
        for key in group_keys:
            mean, ci_low, ci_high = stats[key][model]
            means.append(mean)
            finite = np.isfinite(mean) and np.isfinite(ci_low) and np.isfinite(ci_high)
            err_lo.append(max(0.0, mean - ci_low) if finite else 0.0)
            err_hi.append(max(0.0, ci_high - mean) if finite else 0.0)

        ax.bar(
            x + offsets[i],
            np.array(means, dtype=float),
            width=bar_width * 0.92,  # keeps a surface gap between adjacent bars
            color=MODEL_COLORS.get(model, "#767676"),
            edgecolor="white" if model == HIGHLIGHT_MODEL else BAR_EDGE_F2,
            linewidth=0.6,
            hatch=MODEL_HATCHES.get(model),
            yerr=np.array([err_lo, err_hi], dtype=float),
            error_kw={
                "elinewidth": 0.8,
                "capsize": 2,
                "capthick": 0.8,
                "ecolor": ERROR_GRAY,
            },
            label=MODEL_DISPLAY.get(model, model),
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        group_labels,
        rotation=rotation,
        ha="right" if rotation else "center",
        rotation_mode="anchor" if rotation else "default",
    )
    ax.set_xlim(-0.6, len(group_keys) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_yticks(list(yticks))
    ax.set_yticklabels([f"{t:.{tick_decimals}f}" for t in yticks])
    ax.set_title(title, loc="left", pad=7.0, fontweight="semibold")
    ax.set_ylabel(ylabel, labelpad=2.0)

    tidy_panel(ax)
    ax.tick_params(axis="x", pad=2.0)
    # The white bar edges would chew the bottom spine into dashes, so it is drawn
    # over the bars instead.
    ax.spines["bottom"].set_visible(False)
    ax.axhline(ylim[0], color=AXIS_GRAY, linewidth=0.7, zorder=4, clip_on=False)
    if DRAW_AXIS_BREAKS and ylim[0] > 0.0:
        draw_axis_break(ax)


def _draw_group_breaks(ax, group_keys: Sequence[str], after: Sequence[str]) -> None:
    for key in after:
        if key not in group_keys:
            continue
        idx = list(group_keys).index(key)
        if idx == len(group_keys) - 1:
            continue
        ax.axvline(idx + 0.5, color=SEPARATOR_GRAY, linewidth=0.7, zorder=1)


def plot_stage_aspect(blocks: Dict[str, Stats], output_base: Path) -> None:
    apply_style()

    fig = plt.figure(figsize=(FIG_WIDTH, FIG2_HEIGHT))
    gs = fig.add_gridspec(
        3, 1, height_ratios=FIG2_HEIGHT_RATIOS, hspace=FIG2_HSPACE
    )
    ax_sim = fig.add_subplot(gs[0, 0])
    ax_beh = fig.add_subplot(gs[1, 0])
    # Centre the five-group alignment panel in the bottom row; the flanking cells
    # stay empty. wspace=0 so the panel is exactly its width ratio.
    bottom = gs[2, 0].subgridspec(1, 3, width_ratios=PANEL_C_WIDTH_RATIOS, wspace=0.0)
    ax_align = fig.add_subplot(bottom[0, 1])

    topic_labels = [TOPIC_LABELS.get(k, k) for k in TOPIC_ORDER]
    aspect_labels = [ASPECT_LABELS.get(k, k) for k in ASPECT_ORDER]

    # (a) Simulation Diversity by stage -- a true zero baseline.
    ylim, yticks = _fit_floor(
        blocks["stage_simulation"], TOPIC_ORDER, STAGE_SIM_YLIM, STAGE_SIM_YTICKS
    )
    draw_grouped_bars(
        ax_sim,
        TOPIC_ORDER,
        topic_labels,
        blocks["stage_simulation"],
        title="(a) Simulation Diversity by Stage",
        ylabel="Simulation Diversity",
        ylim=ylim,
        yticks=yticks,
        tick_decimals=1,
    )
    _draw_group_breaks(ax_sim, TOPIC_ORDER, TOPIC_GROUP_BREAKS_AFTER)

    # (b) Behavior Diversity by stage -- a truncated linear axis (no logit), so
    # the low-scoring risk stages stay visible while the ticks keep their
    # original score values.
    ylim, yticks = _fit_floor(
        blocks["stage_behavior"], TOPIC_ORDER, STAGE_BEH_YLIM, STAGE_BEH_YTICKS
    )
    draw_grouped_bars(
        ax_beh,
        TOPIC_ORDER,
        topic_labels,
        blocks["stage_behavior"],
        title="(b) Behavior Diversity by Stage",
        ylabel="Behavior Diversity",
        ylim=ylim,
        yticks=yticks,
    )
    _draw_group_breaks(ax_beh, TOPIC_ORDER, TOPIC_GROUP_BREAKS_AFTER)

    # (c) Profile Alignment by aspect -- five groups, so a shorter panel.
    ylim, yticks = _fit_floor(
        blocks["aspect"], ASPECT_ORDER, ASPECT_YLIM, ASPECT_YTICKS
    )
    draw_grouped_bars(
        ax_align,
        ASPECT_ORDER,
        aspect_labels,
        blocks["aspect"],
        title="(c) Profile Alignment by Aspect",
        ylabel="Profile Alignment",
        ylim=ylim,
        yticks=yticks,
        rotation=ASPECT_TICK_ROTATION,
    )

    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.10, right=0.99)
    add_shared_legend(fig, bar_legend_handles(MODEL_ORDER, BAR_EDGE_F2), y=1.01)
    save(fig, output_base)


# ============================== Figure 3 =====================================
def plot_runs(
    per_metric: Dict[str, Dict[str, Dict[int, Tuple[float, float, float]]]],
    output_base: Path,
) -> None:
    apply_style()

    fig, axes = plt.subplots(
        1,
        len(RUNS_SPECS),
        figsize=(FIG_WIDTH, FIG3_HEIGHT),
        gridspec_kw={"wspace": 0.40},
    )
    axes = np.atleast_1d(axes)

    for panel_idx, (ax, spec) in enumerate(zip(axes, RUNS_SPECS)):
        title, metric_key, _, ylabel = spec
        stats = per_metric[metric_key]
        bounds: List[float] = []
        runs_seen: List[int] = []

        for model in MODEL_ORDER:  # same order as every other figure
            by_run = stats.get(model)
            if not by_run:
                continue
            runs = sorted(by_run)
            means = np.array([by_run[r][0] for r in runs], dtype=float)
            lows = np.array([by_run[r][1] for r in runs], dtype=float)
            highs = np.array([by_run[r][2] for r in runs], dtype=float)
            bounds.extend(lows.tolist() + highs.tolist())
            runs_seen.extend(runs)

            color = MODEL_COLORS.get(model, "#767676")
            ax.fill_between(
                runs,
                lows,
                highs,
                color=color,
                alpha=RUNS_BAND_ALPHA,
                linewidth=0,
                zorder=2,
            )
            is_focus = model == HIGHLIGHT_MODEL
            ax.plot(
                runs,
                means,
                color=color,
                marker=MODEL_MARKERS.get(model, "o"),
                markevery=RUNS_MARKEVERY,
                markersize=4.0 if is_focus else 3.2,
                markeredgecolor="white",
                markeredgewidth=0.5,
                linewidth=RUNS_LINEWIDTH.get(model, RUNS_LINEWIDTH_DEFAULT),
                zorder=4 if is_focus else 3,
            )

        if not bounds:
            raise RuntimeError(f"No values found for {metric_key}.")
        lo, hi = min(bounds), max(bounds)
        margin = max((hi - lo) * 0.10, 0.01)
        ax.set_ylim(max(0.0, lo - margin), min(1.0, hi + margin))
        ax.set_xlim(min(runs_seen) - 0.4, max(runs_seen) + 0.4)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

        ax.set_title(f"({'abc'[panel_idx]}) {title}", loc="left", pad=7.0,
                     fontweight="semibold")
        ax.set_xlabel(RUNS_XLABEL, labelpad=2.0)
        ax.set_ylabel(ylabel, labelpad=2.0)

        tidy_panel(ax, grid_width=0.7)
        ax.spines["left"].set_bounds(*ax.get_ylim())
        ax.spines["bottom"].set_bounds(*ax.get_xlim())
        ax.tick_params(axis="x", length=2.5, pad=2.0)

    fig.subplots_adjust(top=0.775, bottom=0.20, left=0.085, right=0.995)
    add_shared_legend(fig, line_legend_handles(MODEL_ORDER), y=1.02)
    save(fig, output_base)


# ============================== Figure 4 =====================================
def plot_fixattr(
    per_metric: Dict[str, Dict[str, Dict[int, Tuple[float, float, float]]]],
    output_base: Path,
) -> None:
    apply_style()

    fig, axes = plt.subplots(
        1,
        len(FIXATTR_SPECS),
        figsize=(FIG_WIDTH, FIG4_HEIGHT),
        gridspec_kw={"wspace": 0.18},
    )
    axes = np.atleast_1d(axes)

    for panel_idx, (ax, spec) in enumerate(zip(axes, FIXATTR_SPECS)):
        title, section, _, _, ylabel, ylim, yticks = spec
        stats = per_metric[section]
        ks_seen: List[int] = []

        for model in MODEL_ORDER:  # same order as every other figure
            by_k = stats.get(model)
            if not by_k:
                continue
            ks = sorted(by_k)
            means = np.array([by_k[k][0] for k in ks], dtype=float)
            lows = np.array([by_k[k][1] for k in ks], dtype=float)
            highs = np.array([by_k[k][2] for k in ks], dtype=float)
            ks_seen.extend(ks)

            color = MODEL_COLORS.get(model, "#767676")
            ax.fill_between(
                ks, lows, highs, color=color, alpha=FIXATTR_BAND_ALPHA, linewidth=0, zorder=2
            )
            is_focus = model == HIGHLIGHT_MODEL
            ax.plot(
                ks,
                means,
                color=color,
                marker=MODEL_MARKERS.get(model, "o"),
                markevery=1 if is_focus else 2,  # Angel keeps every point
                markersize=FIXATTR_MARKERSIZE.get(model, FIXATTR_MARKERSIZE_DEFAULT),
                markeredgecolor="white",
                markeredgewidth=0.5,
                linewidth=FIXATTR_LINEWIDTH.get(model, FIXATTR_LINEWIDTH_DEFAULT),
                zorder=4 if is_focus else 3,
            )

        if not ks_seen:
            raise RuntimeError(f"No values found for {section}.")
        ax.set_ylim(*ylim)
        ax.set_yticks(list(yticks))
        ax.set_xlim(min(ks_seen) - 0.5, max(ks_seen) + 0.5)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))

        ax.set_title(f"({'abc'[panel_idx]}) {title}", loc="left", pad=7.0,
                     fontweight="semibold")
        ax.set_xlabel(FIXATTR_XLABEL, labelpad=2.0)
        ax.set_ylabel(ylabel, labelpad=2.0)

        tidy_panel(ax, grid_width=0.6)
        ax.spines["left"].set_bounds(*ax.get_ylim())
        ax.spines["bottom"].set_bounds(*ax.get_xlim())
        ax.tick_params(axis="x", length=2.5, pad=2.0)

    fig.subplots_adjust(top=0.785, bottom=0.19, left=0.082, right=0.995)
    add_shared_legend(fig, line_legend_handles(MODEL_ORDER), y=1.02)
    save(fig, output_base)


# ============================== output =======================================
def save(fig, output_base: Path) -> None:
    """Write PDF + PNG, cropped just past the shared legend.

    The legend is anchored above the canvas, so a tight bbox is what keeps it in
    the page. The crop trims a few hundredths of an inch off the nominal size --
    see the sizing note in the module docstring.
    """
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02
    )
    plt.close(fig)


def report(blocks: Dict[str, Stats]) -> None:
    sections = [
        ("Overall (Figure 1)", blocks["overall"], {k: t for t, k, *_ in OVERALL_SPECS}),
        ("Simulation Diversity by stage (Fig 2a)", blocks["stage_simulation"], TOPIC_LABELS),
        ("Behavior Diversity by stage (Fig 2b)", blocks["stage_behavior"], TOPIC_LABELS),
        ("Profile Alignment by aspect (Fig 2c)", blocks["aspect"], ASPECT_LABELS),
    ]
    for heading, stats, labels in sections:
        print(f"\n=== {heading} ===")
        for row_key, per_model in stats.items():
            print(f"  {labels.get(row_key, row_key)}")
            for model in MODEL_ORDER:
                mean, ci_low, ci_high = per_model[model]
                print(
                    f"    {MODEL_DISPLAY.get(model, model):14s} "
                    f"mean={mean:.4f}  ci95=[{ci_low:.4f}, {ci_high:.4f}]"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=layout.CLEAN_DIR,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=layout.FIG_DIR,
    )
    parser.add_argument("--run-number", type=int, default=4)
    parser.add_argument(
        "--runs-input-glob",
        default=str(layout.RESULTS_DIR / "*_agenda_runs25.metrics.run*.json"),
        help="Per-run metric files behind Figure 3.",
    )
    parser.add_argument(
        "--fixattr-input-glob",
        default=str(layout.FIXATTR_DIR / "fixattr_variant_*.metrics.run4.json"),
        help="Fixed-attribute variant metric files behind Figure 4.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-adjust-by-informative-run-ratio",
        action="store_true",
        help="Use raw per-stage scores instead of scores * informative_run_ratio.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=["overall", "stage_aspect", "runs", "fixattr"],
        choices=["overall", "stage_aspect", "runs", "fixattr"],
    )
    parser.add_argument(
        "--print-stats", action="store_true", help="Dump every mean and CI to stdout."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    docs = load_docs(args.clean_dir, args.run_number)
    blocks = compute_all_stats(
        docs,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        adjust_by_ratio=not args.no_adjust_by_informative_run_ratio,
    )

    if "overall" in args.figures:
        base = args.out_dir / "overall_model_performance"
        plot_overall(blocks["overall"], base)
        print(f"[figure 1] saved {base.with_suffix('.pdf')}")
    if "stage_aspect" in args.figures:
        base = args.out_dir / "stage_aspect_performance"
        plot_stage_aspect(blocks, base)
        print(f"[figure 2] saved {base.with_suffix('.pdf')}")
    if "runs" in args.figures:
        per_metric = {
            metric_key: collect_run_stats(
                args.runs_input_glob, metric_key, value_field, args.n_bootstrap, args.seed
            )
            for _, metric_key, value_field, _ in RUNS_SPECS
        }
        base = args.out_dir / "diversity_by_runs"
        plot_runs(per_metric, base)
        print(f"[figure 3] saved {base.with_suffix('.pdf')}")
    if "fixattr" in args.figures:
        per_section = {
            section: collect_fixattr_stats(
                args.fixattr_input_glob,
                section,
                field,
                skip_unscored,
                FIXATTR_N_BOOTSTRAP,
                args.seed,
            )
            for _, section, field, skip_unscored, *_ in FIXATTR_SPECS
        }
        base = args.out_dir / "diversity_by_fixed_attributes"
        plot_fixattr(per_section, base)
        print(f"[figure 4] saved {base.with_suffix('.pdf')}")

    if args.print_stats:
        report(blocks)


if __name__ == "__main__":
    main()
