#!/usr/bin/env python3
"""Combine auto-attack and reframe dot+CI plots with one shared y-scale."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = ["gemini-3-pro", "gpt4o", "claude-4.5-opus"]
DISPLAY_MODEL = {
    "gemini-3-pro": "gemini-3-pro",
    "gpt4o": "gpt-4o",
    "claude-4.5-opus": "claude-4.5-opus",
}


def load_summary(path: Path) -> Dict[str, Dict[str, Tuple[float, float]]]:
    data: Dict[str, Dict[str, Tuple[float, float]]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = row["model"]
            metric = row["metric"]
            mean_value = float(row["mean_from_md"])
            ci95 = float(row["ci95"])
            data.setdefault(model, {})[metric] = (mean_value, ci95)
    return data


def extract_arrays(summary: Dict[str, Dict[str, Tuple[float, float]]]) -> Tuple[List[float], List[float], List[float], List[float]]:
    risk_means: List[float] = []
    risk_ci: List[float] = []
    safety_means: List[float] = []
    safety_ci: List[float] = []

    for model in MODEL_ORDER:
        if model not in summary:
            raise ValueError(f"Missing model '{model}' in summary")
        metrics = summary[model]
        if "Risk" not in metrics or "Safety" not in metrics:
            raise ValueError(f"Missing Risk/Safety for model '{model}'")
        r_mean, r_ci = metrics["Risk"]
        s_mean, s_ci = metrics["Safety"]
        risk_means.append(r_mean)
        risk_ci.append(r_ci)
        safety_means.append(s_mean)
        safety_ci.append(s_ci)
    return risk_means, risk_ci, safety_means, safety_ci


def annotate_scores(
    ax: plt.Axes,
    x_vals: np.ndarray,
    means: List[float],
    cis: List[float],
    color: str,
    ylim_top: float,
) -> None:
    y_offset = ylim_top * 0.010
    for x, mean_val, ci_val in zip(x_vals, means, cis):
        y = mean_val + ci_val + y_offset
        ax.text(
            x,
            y,
            f"{mean_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.78),
        )


def draw_combined_axis(
    ax: plt.Axes,
    auto_risk: List[float],
    auto_risk_ci: List[float],
    auto_safety: List[float],
    auto_safety_ci: List[float],
    ref_risk: List[float],
    ref_risk_ci: List[float],
    ref_safety: List[float],
    ref_safety_ci: List[float],
    ylim_top: float,
) -> None:
    # One continuous axis with two sections and a small gap.
    x_auto = np.array([0.0, 1.0, 2.0])
    x_ref = np.array([3.4, 4.4, 5.4])
    offset = 0.17

    ax.errorbar(
        x_auto - offset,
        auto_risk,
        yerr=auto_risk_ci,
        fmt="o",
        markersize=10,
        capsize=7,
        linewidth=2.0,
        color="#d34a36",
        markerfacecolor="#f09c91",
        markeredgewidth=2.0,
        label="Risk Composite",
    )
    ax.errorbar(
        x_auto + offset,
        auto_safety,
        yerr=auto_safety_ci,
        fmt="o",
        markersize=10,
        capsize=7,
        linewidth=2.0,
        color="#1182cc",
        markerfacecolor="#8fceee",
        markeredgewidth=2.0,
        label="Safety Composite",
    )
    ax.errorbar(
        x_ref - offset,
        ref_risk,
        yerr=ref_risk_ci,
        fmt="o",
        markersize=10,
        capsize=7,
        linewidth=2.0,
        color="#d34a36",
        markerfacecolor="#f09c91",
        markeredgewidth=2.0,
    )
    ax.errorbar(
        x_ref + offset,
        ref_safety,
        yerr=ref_safety_ci,
        fmt="o",
        markersize=10,
        capsize=7,
        linewidth=2.0,
        color="#1182cc",
        markerfacecolor="#8fceee",
        markeredgewidth=2.0,
    )

    annotate_scores(ax, x_auto - offset, auto_risk, auto_risk_ci, "#8b2d1f", ylim_top)
    annotate_scores(ax, x_auto + offset, auto_safety, auto_safety_ci, "#0a5f95", ylim_top)
    annotate_scores(ax, x_ref - offset, ref_risk, ref_risk_ci, "#8b2d1f", ylim_top)
    annotate_scores(ax, x_ref + offset, ref_safety, ref_safety_ci, "#0a5f95", ylim_top)

    xticks = np.concatenate([x_auto, x_ref])
    xlabels = [DISPLAY_MODEL[m] for m in MODEL_ORDER] + [DISPLAY_MODEL[m] for m in MODEL_ORDER]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    ax.axvspan(-0.6, 2.6, color="#f8fafc", zorder=0)
    ax.axvspan(2.8, 6.0, color="#fcfcfd", zorder=0)
    ax.axvline(2.7, color="#8c8c8c", linestyle="--", linewidth=1.3, alpha=0.85)

    ax.text(1.0, ylim_top * 0.98, "Auto-generation", ha="center", va="top", fontsize=22, fontweight="bold", color="#1f2937")
    ax.text(4.4, ylim_top * 0.98, "Reframe", ha="center", va="top", fontsize=22, fontweight="bold", color="#1f2937")

    ax.set_xlim(-0.7, 6.1)
    ax.grid(axis="y", alpha=0.28, linewidth=1.0, color="#94a3b8")
    ax.set_xlabel("Model", fontsize=28, fontweight="bold", labelpad=8)
    ax.tick_params(axis="both", labelsize=24)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auto-summary",
        type=Path,
        default=Path("outputs/safety_exp/fig/auto_attack_risk_safety_ci_summary.csv"),
    )
    parser.add_argument(
        "--reframe-summary",
        type=Path,
        default=Path("outputs/safety_exp/fig/reframe_risk_safety_ci_summary.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/safety_exp/fig/combined_auto_reframe_risk_safety_dot_ci.png"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    auto = load_summary(args.auto_summary)
    reframe = load_summary(args.reframe_summary)

    auto_risk, auto_risk_ci, auto_safety, auto_safety_ci = extract_arrays(auto)
    ref_risk, ref_risk_ci, ref_safety, ref_safety_ci = extract_arrays(reframe)

    ymax = max(
        max(m + c for m, c in zip(auto_risk, auto_risk_ci)),
        max(m + c for m, c in zip(auto_safety, auto_safety_ci)),
        max(m + c for m, c in zip(ref_risk, ref_risk_ci)),
        max(m + c for m, c in zip(ref_safety, ref_safety_ci)),
    )
    ylim_top = ymax * 1.08

    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.titlesize": 34,
            "axes.labelsize": 28,
            "xtick.labelsize": 24,
            "ytick.labelsize": 24,
            "legend.fontsize": 20,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(1, 1, figsize=(24, 10))
    draw_combined_axis(
        ax,
        auto_risk,
        auto_risk_ci,
        auto_safety,
        auto_safety_ci,
        ref_risk,
        ref_risk_ci,
        ref_safety,
        ref_safety_ci,
        ylim_top,
    )

    ax.set_ylim(0.0, ylim_top)
    ax.set_ylabel("Score", fontsize=28, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.83),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.92,
        edgecolor="#d1d5db",
        borderpad=0.6,
        handlelength=1.8,
    )
    fig.suptitle(
        "Mean Risk and Safety Composite Scores by Model",
        fontsize=42,
        fontweight="bold",
        y=0.895,
    )

    fig.tight_layout(pad=0.8, rect=[0.0, 0.0, 1.0, 0.77])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
