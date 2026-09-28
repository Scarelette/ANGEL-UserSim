#!/usr/bin/env python3
"""Create per-model auto-attack Overall Codebook Pattern charts with 95% CI."""

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

MODEL_DISPLAY_NAME: Dict[str, str] = {
    "gpt4o": "GPT-4o",
    "gemini-3-pro": "Gemini-3-pro",
    "claude-4.5-opus": "Claude-4.5-opus",
}


def parse_overall_table(report_path: Path) -> List[Tuple[str, float]]:
    lines = report_path.read_text(encoding="utf-8").splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "## Overall Codebook Pattern":
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(f"Could not find 'Overall Codebook Pattern' in {report_path}")

    entries: List[Tuple[str, float]] = []
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
        if not match:
            continue
        metric = match.group(1)
        mean_val = float(match.group(2))
        entries.append((metric, mean_val))

    if not entries:
        raise ValueError(f"No metric rows parsed from overall table in {report_path}")
    return entries

def infer_mode_from_report(report_path: Path) -> str:
    path = str(report_path).replace("\\", "/")
    if "/reframe/" in path:
        return "reframe"
    return "auto_attack"


def read_metric_values(result_dir: Path, csv_col: str, mode: str) -> List[float]:
    csv_files = sorted(result_dir.glob(f"codebook_judge_long_full_{mode}_profile*.csv"))
    if not csv_files:
        # Some older runs (e.g., gpt4o reframe) use this legacy naming without mode token.
        csv_files = sorted(result_dir.glob("codebook_judge_long_full_profile*.csv"))
    if not csv_files:
        raise ValueError(f"No judge CSV files found in {result_dir} for mode={mode}")

    values: List[float] = []
    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                values.append(float(row[csv_col]))
    if not values:
        raise ValueError(f"No rows found for column {csv_col} in {result_dir}")
    return values


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def std_sample(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((x - mu) ** 2 for x in values) / (len(values) - 1))


def ci95(values: List[float]) -> float:
    return 1.96 * std_sample(values) / math.sqrt(len(values))


def pretty_model_name(model_dir: str) -> str:
    return MODEL_DISPLAY_NAME.get(model_dir, model_dir)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", text.lower()).strip("-")


def style_cfg(style: str) -> Dict[str, object]:
    if style == "slide":
        return {
            "rc": {
                "font.size": 14,
                "axes.titlesize": 24,
                "axes.labelsize": 20,
                "xtick.labelsize": 17,
                "ytick.labelsize": 16,
                "figure.facecolor": "white",
                "axes.facecolor": "white",
            },
            "figsize": (20, 10),
            "bar_color": "#7fb3d5",
            "bar_edge": "#1f77b4",
            "bar_lw": 1.5,
            "capsize": 5,
            "value_fontsize": 14,
            "rotation": 0,
            "grid_alpha": 0.25,
            "title_suffix": "",
            "dot_color": "#1f77b4",
            "dot_face": "#7fb3d5",
            "dot_markersize": 8.5,
            "dot_lw": 1.8,
            "dot_line_lw": 1.8,
            "grid_color": "#999999",
            "spine_color": "#000000",
            "value_color": "#111111",
        }
    return {
        "rc": {
            "font.size": 12,
            "axes.titlesize": 20,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        },
        "figsize": (17, 8),
        "bar_color": "#7fb3d5",
        "bar_edge": "#1f77b4",
        "bar_lw": 1.3,
        "capsize": 5,
        "value_fontsize": 11,
        "rotation": 0,
        "grid_alpha": 0.25,
        "title_suffix": "",
        "dot_color": "#1f77b4",
        "dot_face": "#7fb3d5",
        "dot_markersize": 7.5,
        "dot_lw": 1.6,
        "dot_line_lw": 1.6,
        "grid_color": "#999999",
        "spine_color": "#000000",
        "value_color": "#111111",
    }


def plot_model(
    report_path: Path,
    out_dir: Path,
    style: str,
    dpi: int,
    filename_suffix: str,
    chart_type: str,
) -> Path:
    model_dir = report_path.parent.parent.name
    model_name = pretty_model_name(model_dir)
    result_dir = report_path.parent.parent
    mode = infer_mode_from_report(report_path)
    metric_means = parse_overall_table(report_path)

    metrics: List[str] = []
    means_from_report: List[float] = []
    ci_values: List[float] = []

    for metric_name, mean_val in metric_means:
        if metric_name not in METRIC_TO_CSV_COLUMN:
            continue
        csv_col = METRIC_TO_CSV_COLUMN[metric_name]
        vals = read_metric_values(result_dir, csv_col, mode=mode)
        metrics.append(metric_name)
        means_from_report.append(mean_val)
        ci_values.append(ci95(vals))

    if not metrics:
        raise ValueError(f"No supported metrics found in {report_path}")

    x = np.arange(len(metrics))
    display_metrics = [METRIC_DISPLAY_NAME.get(metric, metric) for metric in metrics]
    ymax = max(m + c for m, c in zip(means_from_report, ci_values))
    ylim_top = ymax * 1.12

    cfg = style_cfg(style)
    plt.rcParams.update(cfg["rc"])  # type: ignore[arg-type]
    fig, ax = plt.subplots(figsize=cfg["figsize"])  # type: ignore[arg-type]
    ax.set_facecolor("white")
    if chart_type == "bar":
        bars = ax.bar(
            x,
            means_from_report,
            yerr=ci_values,
            capsize=cfg["capsize"],  # type: ignore[arg-type]
            color=cfg["bar_color"],  # type: ignore[arg-type]
            edgecolor=cfg["bar_edge"],  # type: ignore[arg-type]
            linewidth=cfg["bar_lw"],  # type: ignore[arg-type]
        )

        for i, bar in enumerate(bars):
            y = means_from_report[i] + ci_values[i] + ylim_top * 0.008
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{means_from_report[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=cfg["value_fontsize"],  # type: ignore[arg-type]
                color=cfg["value_color"],  # type: ignore[arg-type]
            )
    else:
        for xi, mean_val, ci_val in zip(x, means_from_report, ci_values):
            ax.errorbar(
                [xi],
                [mean_val],
                yerr=ci_val,
                fmt="o",
                markersize=cfg["dot_markersize"],  # type: ignore[arg-type]
                capsize=cfg["capsize"],  # type: ignore[arg-type]
                color=cfg["dot_color"],  # type: ignore[arg-type]
                markerfacecolor=cfg["dot_face"],  # type: ignore[arg-type]
                markeredgecolor=cfg["dot_color"],  # type: ignore[arg-type]
                markeredgewidth=cfg["dot_lw"],  # type: ignore[arg-type]
                linewidth=cfg["dot_line_lw"],  # type: ignore[arg-type]
            )
        for i, xi in enumerate(x):
            y = means_from_report[i] + ci_values[i] + ylim_top * 0.012
            ax.text(
                xi,
                y,
                f"{means_from_report[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=cfg["value_fontsize"],  # type: ignore[arg-type]
                color=cfg["value_color"],  # type: ignore[arg-type]
            )

    ax.set_xticks(x)
    ax.set_xticklabels(display_metrics, rotation=cfg["rotation"], ha="center", fontweight="normal")  # type: ignore[arg-type]
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_ylim(0.0, ylim_top)
    ax.set_title(f"Score on Codebook Patterns ({model_name}){cfg['title_suffix']}", pad=16, fontweight="bold")
    ax.grid(axis="y", alpha=cfg["grid_alpha"], color=cfg["grid_color"])  # type: ignore[arg-type]
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(True)
    ax.spines["left"].set_color(cfg["spine_color"])  # type: ignore[arg-type]
    ax.spines["bottom"].set_color(cfg["spine_color"])  # type: ignore[arg-type]

    fig.tight_layout(pad=0.8)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{filename_suffix}" if filename_suffix else ""
    out_path = out_dir / f"{slug(model_dir)}_{mode}_overall_codebook_pattern_{chart_type}_ci{suffix}.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-glob",
        default="outputs/safety_exp/results/auto_attack/*/summary/auto_attack_codebook_report.md",
        help="Glob pattern for auto-attack summary report files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/safety_exp/fig/fig2"),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--style",
        choices=["default", "slide"],
        default="default",
        help="Figure style preset.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PNG dpi for output figures.",
    )
    parser.add_argument(
        "--filename-suffix",
        default="",
        help="Optional suffix appended to output file names (without leading underscore).",
    )
    parser.add_argument(
        "--chart-type",
        choices=["bar", "dot", "both"],
        default="bar",
        help="Chart type to generate.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_paths = sorted(Path().glob(args.reports_glob))
    if not report_paths:
        raise SystemExit(f"No report files found with glob: {args.reports_glob}")

    chart_types = ["bar", "dot"] if args.chart_type == "both" else [args.chart_type]
    for report_path in report_paths:
        for chart_type in chart_types:
            out_path = plot_model(
                report_path,
                args.out_dir,
                style=args.style,
                dpi=args.dpi,
                filename_suffix=args.filename_suffix,
                chart_type=chart_type,
            )
            print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
