#!/usr/bin/env python3
"""Combine auto-generation (left) and reframe (right) overall metric plots per model."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


METRIC_TO_CSV_COLUMN: Dict[str, str] = {
    "Risk": "Risk",
    "Safety": "Safety",
    "SinglePrompt": "SinglePrompt",
    "Sycophancy": "Sycophancy",
    "Validation": "Validation",
    "Elaboration": "Elaboration",
    "Behavioral Advice": "Behavioral",
    "Misrepresentation": "Misrepresentation",
    "Reality Testing": "Reality",
    "Concern for Wellbeing": "Concern",
    "Referral": "Referral",
    "De-escalation": "DeEscalation",
    "Relational Warmth": "Warmth",
}

METRIC_DISPLAY_NAME: Dict[str, str] = {
    "Risk": "Risk",
    "Safety": "Safety",
    "SinglePrompt": "Single",
    "Sycophancy": "Sycoph.",
    "Validation": "Valid.",
    "Elaboration": "Elab.",
    "Behavioral Advice": "Behav. Adv.",
    "Misrepresentation": "Misrep.",
    "Reality Testing": "Reality",
    "Concern for Wellbeing": "Wellbeing",
    "Referral": "Referral",
    "De-escalation": "De-esc.",
    "Relational Warmth": "Warmth",
}

# Pink + green palette.
AUTO_FILL = "#f6a1a1"
AUTO_EDGE = "#c24141"
REF_FILL = "#9ad98c"
REF_EDGE = "#2f855a"
TEXT_DARK = "#1f2937"


def pretty_model_name(model_dir: str) -> str:
    if model_dir == "gpt4o":
        return "gpt-4o"
    return model_dir


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-")


def parse_overall_table(report_path: Path) -> List[Tuple[str, float]]:
    lines = report_path.read_text(encoding="utf-8").splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "## Overall Codebook Pattern":
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(f"Could not find 'Overall Codebook Pattern' in {report_path}")

    rows: List[Tuple[str, float]] = []
    row_pattern = re.compile(r"^\|\s*(.+?)\s*\|\s*([0-9]*\.?[0-9]+)\s*\|$")
    for line in lines[start_idx + 1 :]:
        if line.startswith("## "):
            break
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        if "Metric" in stripped and "Mean" in stripped:
            continue
        if set(stripped.replace("|", "").strip()) <= {"-", ":"}:
            continue
        match = row_pattern.match(stripped)
        if match:
            metric_name = match.group(1)
            mean_val = float(match.group(2))
            rows.append((metric_name, mean_val))
    if not rows:
        raise ValueError(f"No metric rows parsed from {report_path}")
    return rows


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def std_sample(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def ci95(values: List[float]) -> float:
    return 1.96 * std_sample(values) / math.sqrt(len(values))


def read_metric_values(result_dir: Path, csv_col: str, mode: str) -> List[float]:
    if mode == "auto_attack":
        patterns = ["codebook_judge_long_full_auto_attack_profile*.csv"]
    else:
        patterns = [
            "codebook_judge_long_full_reframe_profile*.csv",
            "codebook_judge_long_full_profile*.csv",  # legacy gpt4o naming
        ]

    csv_files: List[Path] = []
    for pattern in patterns:
        csv_files = sorted(result_dir.glob(pattern))
        if csv_files:
            break
    if not csv_files:
        raise ValueError(f"No CSV files found in {result_dir} for mode={mode}")

    values: List[float] = []
    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                values.append(float(row[csv_col]))
    if not values:
        raise ValueError(f"No values for column {csv_col} in {result_dir}")
    return values


def load_mode_metrics(
    report_path: Path,
    result_dir: Path,
    mode: str,
) -> Tuple[List[str], List[float], List[float]]:
    table = parse_overall_table(report_path)
    metric_names: List[str] = []
    means_from_report: List[float] = []
    ci_values: List[float] = []
    for metric_name, mean_val in table:
        if metric_name not in METRIC_TO_CSV_COLUMN:
            continue
        csv_col = METRIC_TO_CSV_COLUMN[metric_name]
        vals = read_metric_values(result_dir, csv_col, mode=mode)
        metric_names.append(metric_name)
        means_from_report.append(mean_val)
        ci_values.append(ci95(vals))
    if not metric_names:
        raise ValueError(f"No supported metrics found in {report_path}")
    return metric_names, means_from_report, ci_values


def annotate(
    ax: plt.Axes,
    xvals: np.ndarray,
    means: List[float],
    cis: List[float],
    ylim_top: float,
) -> None:
    for x, m, c in zip(xvals, means, cis):
        ax.text(
            x,
            m + c + ylim_top * 0.01,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="semibold",
            color=TEXT_DARK,
        )


def draw_combined_bar(
    model_name: str,
    metrics: List[str],
    auto_means: List[float],
    auto_ci: List[float],
    ref_means: List[float],
    ref_ci: List[float],
    out_path: Path,
    dpi: int,
) -> None:
    n = len(metrics)
    x_left = np.arange(n, dtype=float)
    gap = 0.55
    x_right = x_left + n + gap
    divider_x = n - 0.5 + gap / 2

    ymax = max(
        max(m + c for m, c in zip(auto_means, auto_ci)),
        max(m + c for m, c in zip(ref_means, ref_ci)),
    )
    ylim_top = ymax * 1.11

    display_metrics = [METRIC_DISPLAY_NAME.get(m, m) for m in metrics]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 34,
            "axes.labelsize": 22,
            "xtick.labelsize": 16,
            "ytick.labelsize": 15,
        }
    )

    fig, ax = plt.subplots(figsize=(28, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfdff")
    bar_width = 0.8
    bars_auto = ax.bar(
        x_left,
        auto_means,
        width=bar_width,
        yerr=auto_ci,
        capsize=4,
        color=AUTO_FILL,
        edgecolor=AUTO_EDGE,
        linewidth=1.4,
    )
    bars_ref = ax.bar(
        x_right,
        ref_means,
        width=bar_width,
        yerr=ref_ci,
        capsize=4,
        color=REF_FILL,
        edgecolor=REF_EDGE,
        linewidth=1.4,
    )

    annotate(ax, np.array([b.get_x() + b.get_width() / 2 for b in bars_auto]), auto_means, auto_ci, ylim_top)
    annotate(ax, np.array([b.get_x() + b.get_width() / 2 for b in bars_ref]), ref_means, ref_ci, ylim_top)

    xticks = np.concatenate([x_left, x_right])
    xlabels = display_metrics + display_metrics
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=24, ha="right")

    ax.axvspan(-0.75, n - 0.35, color="#fff1f2", zorder=0, alpha=0.6)
    ax.axvspan(n + gap - 0.65, n + gap + n - 0.25, color="#f0fdf4", zorder=0, alpha=0.6)
    ax.axvline(divider_x, color="#64748b", linestyle=(0, (4, 3)), linewidth=1.6, alpha=0.9)
    ax.text((x_left[0] + x_left[-1]) / 2, ylim_top * 0.985, "Auto-generation", ha="center", va="top", fontsize=23, fontweight="bold", color=TEXT_DARK)
    ax.text((x_right[0] + x_right[-1]) / 2, ylim_top * 0.985, "Reframe", ha="center", va="top", fontsize=23, fontweight="bold", color=TEXT_DARK)

    ax.set_xlim(-0.95, x_right[-1] + 0.95)
    ax.set_ylim(0.0, ylim_top)
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.grid(axis="y", alpha=0.2, linewidth=1.1, color="#94a3b8")
    ax.set_axisbelow(True)
    ax.set_title(f"Safety and Risk Score ({model_name})", pad=18, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")
    fig.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def draw_combined_dot(
    model_name: str,
    metrics: List[str],
    auto_means: List[float],
    auto_ci: List[float],
    ref_means: List[float],
    ref_ci: List[float],
    out_path: Path,
    dpi: int,
) -> None:
    n = len(metrics)
    x_left = np.arange(n, dtype=float)
    gap = 0.55
    x_right = x_left + n + gap
    divider_x = n - 0.5 + gap / 2

    ymax = max(
        max(m + c for m, c in zip(auto_means, auto_ci)),
        max(m + c for m, c in zip(ref_means, ref_ci)),
    )
    ylim_top = ymax * 1.11

    display_metrics = [METRIC_DISPLAY_NAME.get(m, m) for m in metrics]

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 34,
            "axes.labelsize": 22,
            "xtick.labelsize": 16,
            "ytick.labelsize": 15,
        }
    )

    fig, ax = plt.subplots(figsize=(28, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fcfdff")
    ax.errorbar(
        x_left,
        auto_means,
        yerr=auto_ci,
        fmt="o",
        markersize=8.5,
        capsize=4,
        linewidth=1.8,
        color=AUTO_EDGE,
        markerfacecolor=AUTO_FILL,
        markeredgewidth=1.8,
    )
    ax.errorbar(
        x_right,
        ref_means,
        yerr=ref_ci,
        fmt="o",
        markersize=8.5,
        capsize=4,
        linewidth=1.8,
        color=REF_EDGE,
        markerfacecolor=REF_FILL,
        markeredgewidth=1.8,
    )

    annotate(ax, x_left, auto_means, auto_ci, ylim_top)
    annotate(ax, x_right, ref_means, ref_ci, ylim_top)

    xticks = np.concatenate([x_left, x_right])
    xlabels = display_metrics + display_metrics
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=24, ha="right")

    ax.axvspan(-0.75, n - 0.35, color="#fff1f2", zorder=0, alpha=0.6)
    ax.axvspan(n + gap - 0.65, n + gap + n - 0.25, color="#f0fdf4", zorder=0, alpha=0.6)
    ax.axvline(divider_x, color="#64748b", linestyle=(0, (4, 3)), linewidth=1.6, alpha=0.9)
    ax.text((x_left[0] + x_left[-1]) / 2, ylim_top * 0.985, "Auto-generation", ha="center", va="top", fontsize=23, fontweight="bold", color=TEXT_DARK)
    ax.text((x_right[0] + x_right[-1]) / 2, ylim_top * 0.985, "Reframe", ha="center", va="top", fontsize=23, fontweight="bold", color=TEXT_DARK)

    ax.set_xlim(-0.95, x_right[-1] + 0.95)
    ax.set_ylim(0.0, ylim_top)
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.grid(axis="y", alpha=0.2, linewidth=1.1, color="#94a3b8")
    ax.set_axisbelow(True)
    ax.set_title(f"Safety and Risk Score ({model_name})", pad=18, fontweight="bold")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")
    fig.tight_layout(pad=0.6)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auto-root",
        type=Path,
        default=Path("outputs/safety_exp/results/auto_attack"),
    )
    parser.add_argument(
        "--reframe-root",
        type=Path,
        default=Path("outputs/safety_exp/results/reframe"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/safety_exp/fig/fig2"),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    auto_reports = sorted(args.auto_root.glob("*/summary/auto_attack_codebook_report.md"))
    reframe_reports = sorted(args.reframe_root.glob("*/summary/reframe_codebook_report.md"))
    auto_by_model = {p.parent.parent.name: p for p in auto_reports}
    reframe_by_model = {p.parent.parent.name: p for p in reframe_reports}
    common_models = sorted(set(auto_by_model) & set(reframe_by_model))
    if not common_models:
        raise SystemExit("No overlapping models found between auto_attack and reframe summary reports.")

    for model in common_models:
        auto_report = auto_by_model[model]
        ref_report = reframe_by_model[model]
        auto_dir = auto_report.parent.parent
        ref_dir = ref_report.parent.parent

        auto_metrics, auto_means, auto_ci = load_mode_metrics(auto_report, auto_dir, "auto_attack")
        ref_metrics, ref_means, ref_ci = load_mode_metrics(ref_report, ref_dir, "reframe")

        if auto_metrics != ref_metrics:
            raise ValueError(f"Metric order mismatch for model={model}")

        model_pretty = pretty_model_name(model)
        bar_out = args.out_dir / f"{slug(model)}_auto_vs_reframe_overall_codebook_pattern_bar_ci.png"
        dot_out = args.out_dir / f"{slug(model)}_auto_vs_reframe_overall_codebook_pattern_dot_ci.png"

        draw_combined_bar(
            model_pretty,
            auto_metrics,
            auto_means,
            auto_ci,
            ref_means,
            ref_ci,
            bar_out,
            args.dpi,
        )
        draw_combined_dot(
            model_pretty,
            auto_metrics,
            auto_means,
            auto_ci,
            ref_means,
            ref_ci,
            dot_out,
            args.dpi,
        )

        print(bar_out)
        print(dot_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
