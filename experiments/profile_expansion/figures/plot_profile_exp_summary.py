#!/usr/bin/env python3
"""Polished 3-panel summary figure for the profile-expansion experiment.

Panels (left to right): Simulation Diversity (min_distance_diversity),
Behavior Diversity, Profile Alignment. Each panel shows the per-model mean
with a 95% bootstrap CI.

This is the presentation-quality version of
``plot_run_four_diversity_panels_clean.py``: every text element is >= 14pt, and
each model carries a distinct marker shape so identity never rests on color
alone (the project palette is kept unchanged).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


MODEL_ORDER = ["eeyore", "angel", "patient_psi", "roleplay_doh"]

# One type ramp, one knob. BASE is the tick size -- the figure's body text; every
# other element is a small, deliberate step off it, so the panels read as one
# system instead of a pile of unrelated sizes. Change BASE to resize everything.
BASE_FONTSIZE = 22.5

TICK_FONTSIZE = BASE_FONTSIZE  # 22.5
X_TICK_FONTSIZE = BASE_FONTSIZE  # 22.5 -- same as the y ticks, on purpose
VALUE_FONTSIZE = BASE_FONTSIZE * 0.93  # 21   -- subordinate to the ticks
LEGEND_FONTSIZE = BASE_FONTSIZE * 1.02  # 23
AXIS_LABEL_FONTSIZE = BASE_FONTSIZE * 1.07  # 24
TITLE_FONTSIZE = BASE_FONTSIZE * 1.16  # 26   -- leads without shouting

# The model names no longer fit side by side at this size, so they tilt.
X_TICK_ROTATION = 20

# Original project palette, kept as-is by request. These pale fills sit below a
# 3:1 contrast ratio on white, so identity never rests on color alone: each mark
# sits above its named x tick and every model has its own marker shape.
MODEL_COLORS = {
    "eeyore": "#ffbe78",
    "angel": "#1f77b4",
    "patient_psi": "#98df8a",
    "roleplay_doh": "#f28e8c",
}

# Secondary (non-color) encoding of model identity, for print and CVD readers.
MODEL_MARKERS = {
    "eeyore": "o",
    "angel": "D",
    "patient_psi": "s",
    "roleplay_doh": "^",
}

MODEL_DISPLAY_NAMES = {
    "eeyore": "Eeyore",
    "angel": "Angel",
    "patient_psi": "Patient-psi",
    "roleplay_doh": "Roleplay-doh",
}

# The proposed system, emphasised with a bold tick label.
HIGHLIGHT_MODEL = "angel"

INK = "#1a1a1a"
INK_MUTED = "#5a5a5a"
AXIS_GRAY = "#9a9a9a"
GRID_GRAY = "#c8c8c8"

# (panel title, metric key in the combined JSON, per-profile value field)
METRIC_SPECS = [
    ("Simulation Diversity", "min_distance_diversity", "min_distance_diversity"),
    ("Behavior Diversity", "behavior_diversity", "behavior_diversity"),
    ("Profile Alignment", "profile_alignment", "score"),
]

PANEL_TAGS = ["(a)", "(b)", "(c)"]

# Metrics with a fixed (non-auto) y-axis range: (y_min, y_max).
FIXED_YLIM = {
    "profile_alignment": (0.80, 1.00),
}


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": TICK_FONTSIZE,
            "axes.edgecolor": AXIS_GRAY,
            "axes.linewidth": 1.1,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": AXIS_GRAY,
            "ytick.color": AXIS_GRAY,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK_MUTED,
            "xtick.major.size": 0.0,
            "ytick.major.size": 4.0,
            "xtick.major.pad": 8.0,
            "ytick.major.width": 1.1,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,  # editable/embeddable TrueType, not Type-3
            "ps.fonttype": 42,
        }
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())


def _build_metric_blobs(clean_dir: Path, run_number: int) -> Dict[str, dict]:
    by_model: Dict[str, dict] = {}
    for model in MODEL_ORDER:
        path = clean_dir / f"{model}_agenda_runs5.metrics.combined.run{run_number}.json"
        by_model[model] = _read_json(path)
    return by_model


def _extract_metric_values(metrics: dict, metric_name: str, value_field: str) -> List[float]:
    if metric_name == "profile_alignment":
        rows = metrics["profile_alignment"]["by_model_profile"]
        return [float(row[value_field]) for row in rows if row.get(value_field) is not None]
    profiles = metrics[metric_name]["profiles"]
    if metric_name == "behavior_diversity":
        return [float(row[value_field]) for row in profiles if row.get("scored", True)]
    return [float(row[value_field]) for row in profiles if row.get(value_field) is not None]


def _bootstrap_mean_ci(
    values: List[float], n_bootstrap: int, rng: np.random.Generator
) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("Cannot bootstrap an empty set of values.")
    indices = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    sample_means = arr[indices].mean(axis=1)
    low, high = np.percentile(sample_means, [2.5, 97.5])
    return float(arr.mean()), float(low), float(high)


def _compute_stats(
    by_model: Dict[str, dict], models: List[str], n_bootstrap: int, seed: int
) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    rng = np.random.default_rng(seed)
    stats: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for _, metric_name, value_field in METRIC_SPECS:
        stats[metric_name] = {}
        for model in models:
            values = _extract_metric_values(by_model[model], metric_name, value_field)
            stats[metric_name][model] = _bootstrap_mean_ci(values, n_bootstrap, rng)
    return stats


def _auto_ylim(
    stats: Dict[str, Dict[str, Tuple[float, float, float]]], metric_name: str
) -> Tuple[float, float]:
    """Padded data range, leaving headroom for the value labels."""
    if metric_name in FIXED_YLIM:
        return FIXED_YLIM[metric_name]
    lows = [stats[metric_name][m][1] for m in stats[metric_name]]
    highs = [stats[metric_name][m][2] for m in stats[metric_name]]
    lo, hi = min(lows), max(highs)
    pad = (hi - lo) * 0.22 or 0.01
    return max(0.0, lo - pad), min(1.0, hi + pad)


def plot_metrics_dot_ci(
    stats: Dict[str, Dict[str, Tuple[float, float, float]]],
    output_path: Path,
    models: List[str],
) -> None:
    _apply_style()

    ncols = len(METRIC_SPECS)
    fig, axes = plt.subplots(1, ncols, figsize=(6.2 * ncols, 7.2))
    axes = np.atleast_1d(axes)

    x = np.arange(len(models))
    display_models = [MODEL_DISPLAY_NAMES.get(m, m) for m in models]

    for panel_idx, (ax, (title, metric_name, _)) in enumerate(zip(axes, METRIC_SPECS)):
        y_min, y_max = _auto_ylim(stats, metric_name)
        span = y_max - y_min
        y_top = y_max + span * 0.06

        for i, model in enumerate(models):
            mean, ci_low, ci_high = stats[metric_name][model]
            yerr = np.array([[mean - ci_low], [ci_high - mean]])
            color = MODEL_COLORS.get(model, "#767676")
            is_focus = model == HIGHLIGHT_MODEL

            ax.errorbar(
                i,
                mean,
                yerr=yerr,
                fmt="none",
                ecolor=color,
                elinewidth=3.2,
                capsize=9,
                capthick=3.2,
                alpha=0.95,
                zorder=3,
            )
            ax.plot(
                i,
                mean,
                marker=MODEL_MARKERS.get(model, "o"),
                markersize=16 if is_focus else 13,
                color=color,
                markeredgecolor="#2e2e2e",
                markeredgewidth=1.1,  # dark ring: the pale fills need an outline
                linestyle="none",
                zorder=5,
            )
            # Direct value label: one per mark, in muted ink (never series color).
            # Sits above the upper CI cap, or flips below when the panel top is
            # too close (e.g. alignment scores that press against 1.00).
            label_above = (y_top - ci_high) > span * 0.13
            ax.annotate(
                f"{mean:.3f}",
                xy=(i, ci_high if label_above else ci_low),
                xytext=(0, 14 if label_above else -15),
                textcoords="offset points",
                ha="center",
                va="bottom" if label_above else "top",
                fontsize=VALUE_FONTSIZE,
                color=INK_MUTED,
                zorder=6,
            )

        # The tag rides in the title: at this type size a separate corner label
        # would run into it.
        tag = PANEL_TAGS[panel_idx] if panel_idx < len(PANEL_TAGS) else ""
        ax.set_title(
            f"{tag} {title}".strip(),
            fontsize=TITLE_FONTSIZE,
            pad=18,
            fontweight="semibold",
        )
        ax.set_xlim(-0.6, len(models) - 0.4)
        ax.set_ylim(y_min, y_top)
        ax.set_xticks(x)
        ax.set_xticklabels(
            display_models,
            fontsize=X_TICK_FONTSIZE,
            rotation=X_TICK_ROTATION,
            ha="right",
            rotation_mode="anchor",
        )
        for label, model in zip(ax.get_xticklabels(), models):
            if model == HIGHLIGHT_MODEL:
                label.set_fontweight("bold")
                label.set_color(INK)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
        ax.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
        ax.grid(axis="y", color=GRID_GRAY, linestyle=(0, (2, 4)), linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_bounds(*ax.get_ylim())

    axes[0].set_ylabel("Score", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)

    legend_handles = [
        Line2D(
            [],
            [],
            marker=MODEL_MARKERS.get(model, "o"),
            markersize=16 if model == HIGHLIGHT_MODEL else 13,
            color=MODEL_COLORS.get(model, "#767676"),
            markeredgecolor="#2e2e2e",
            markeredgewidth=1.1,
            linestyle="none",
            label=MODEL_DISPLAY_NAMES.get(model, model),
        )
        for model in models
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(models),
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=LEGEND_FONTSIZE,
        handletextpad=0.5,
        columnspacing=2.4,
        labelcolor=INK,
    )

    fig.tight_layout(rect=(0, 0.01, 1, 0.90))
    fig.subplots_adjust(wspace=0.22)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.12)
    fig.savefig(
        output_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.12
    )
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Polished 3-panel profile-expansion summary figure."
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=layout.CLEAN_DIR,
    )
    parser.add_argument("--run-number", type=int, default=4)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=layout.FIG_DIR / "profile_exp_summary.pdf",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    by_model = _build_metric_blobs(args.clean_dir, args.run_number)
    stats = _compute_stats(by_model, MODEL_ORDER, args.n_bootstrap, args.seed)

    plot_metrics_dot_ci(stats=stats, output_path=args.output, models=MODEL_ORDER)

    print(f"Saved figure to: {args.output.with_suffix('.pdf')}")
    print(f"                 {args.output.with_suffix('.png')}")
    print("\nMetric means and 95% bootstrap CIs:")
    for title, metric_name, _ in METRIC_SPECS:
        print(f"\n{title}")
        for model in MODEL_ORDER:
            mean, ci_low, ci_high = stats[metric_name][model]
            print(f"  {model:14s} mean={mean:.6f}  ci95=[{ci_low:.6f}, {ci_high:.6f}]")


if __name__ == "__main__":
    main()
