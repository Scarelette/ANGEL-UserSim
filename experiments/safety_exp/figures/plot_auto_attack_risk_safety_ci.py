#!/usr/bin/env python3
"""Plot mean Risk/Safety with 95% confidence intervals for codebook reports."""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


RISK_PATTERN = re.compile(r"\|\s*Risk\s*\|\s*([0-9]*\.?[0-9]+)\s*\|")
SAFETY_PATTERN = re.compile(r"\|\s*Safety\s*\|\s*([0-9]*\.?[0-9]+)\s*\|")
SOURCE_DIR_PATTERN = re.compile(r"Source directory:\s*`([^`]+)`")


@dataclass
class MetricStats:
    mean_from_md: float
    mean_from_csv: float
    ci95: float
    n: int


@dataclass
class ModelStats:
    model: str
    risk: MetricStats
    safety: MetricStats


def parse_md_summary(report_path: Path) -> Tuple[float, float, Path | None]:
    text = report_path.read_text(encoding="utf-8")
    risk_match = RISK_PATTERN.search(text)
    safety_match = SAFETY_PATTERN.search(text)
    if risk_match is None or safety_match is None:
        raise ValueError(f"Could not parse Risk/Safety means from {report_path}")
    source_match = SOURCE_DIR_PATTERN.search(text)
    source_dir = Path(source_match.group(1)) if source_match else None
    return float(risk_match.group(1)), float(safety_match.group(1)), source_dir


def load_metric_values(result_dir: Path, mode: str) -> Tuple[List[float], List[float]]:
    risk_values: List[float] = []
    safety_values: List[float] = []
    csv_files = sorted(result_dir.glob(f"codebook_judge_long_full_{mode}_profile*.csv"))
    if not csv_files:
        csv_files = sorted(result_dir.glob("codebook_judge_long_full_profile*.csv"))
    if not csv_files:
        raise ValueError(f"No {mode} judge CSV files found in {result_dir}")

    for csv_path in csv_files:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                risk_values.append(float(row["Risk"]))
                safety_values.append(float(row["Safety"]))
    return risk_values, safety_values


def resolve_metric_dir(report_path: Path, mode: str, source_dir_hint: Path | None) -> Path:
    candidates: List[Path] = []
    if source_dir_hint is not None:
        candidates.append(source_dir_hint)
    candidates.append(report_path.parent.parent)

    seen: set[str] = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        has_mode_pattern = any(candidate.glob(f"codebook_judge_long_full_{mode}_profile*.csv"))
        has_generic_pattern = any(candidate.glob("codebook_judge_long_full_profile*.csv"))
        if has_mode_pattern or has_generic_pattern:
            return candidate

    raise ValueError(
        f"No {mode} judge CSV files found for report {report_path}. Checked: "
        + ", ".join(str(p) for p in unique_candidates)
    )


def mean(values: List[float]) -> float:
    return sum(values) / len(values)


def std_sample(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def ci95(values: List[float]) -> float:
    if not values:
        return 0.0
    se = std_sample(values) / math.sqrt(len(values))
    return 1.96 * se


def build_stats(model: str, report_path: Path, mode: str) -> ModelStats:
    risk_mean_md, safety_mean_md, source_dir_hint = parse_md_summary(report_path)
    result_dir = resolve_metric_dir(report_path, mode=mode, source_dir_hint=source_dir_hint)
    risk_values, safety_values = load_metric_values(result_dir, mode=mode)

    risk_stats = MetricStats(
        mean_from_md=risk_mean_md,
        mean_from_csv=mean(risk_values),
        ci95=ci95(risk_values),
        n=len(risk_values),
    )
    safety_stats = MetricStats(
        mean_from_md=safety_mean_md,
        mean_from_csv=mean(safety_values),
        ci95=ci95(safety_values),
        n=len(safety_values),
    )
    return ModelStats(model=model, risk=risk_stats, safety=safety_stats)


def draw_dot_plot(stats: List[ModelStats], output_path: Path, mode: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
        }
    )
    labels = [s.model for s in stats]
    x = np.arange(len(labels), dtype=float)
    risk_means = [s.risk.mean_from_md for s in stats]
    safety_means = [s.safety.mean_from_md for s in stats]
    risk_ci = [s.risk.ci95 for s in stats]
    safety_ci = [s.safety.ci95 for s in stats]

    fig, ax = plt.subplots(figsize=(14, 8))
    offset = 0.14

    ax.errorbar(
        x - offset,
        risk_means,
        yerr=risk_ci,
        fmt="o",
        markersize=8,
        capsize=5,
        linewidth=1.8,
        color="#c0392b",
        label="Mean Risk",
    )
    ax.errorbar(
        x + offset,
        safety_means,
        yerr=safety_ci,
        fmt="o",
        markersize=8,
        capsize=5,
        linewidth=1.8,
        color="#1f77b4",
        label="Mean Safety",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Mean Risk and Safety Score by Model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def draw_bar_plot(stats: List[ModelStats], output_path: Path, mode: str) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "legend.fontsize": 16,
        }
    )
    labels = [s.model for s in stats]
    x = np.arange(len(labels), dtype=float)
    width = 0.34
    risk_means = [s.risk.mean_from_md for s in stats]
    safety_means = [s.safety.mean_from_md for s in stats]
    risk_ci = [s.risk.ci95 for s in stats]
    safety_ci = [s.safety.ci95 for s in stats]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(
        x - width / 2,
        risk_means,
        width,
        yerr=risk_ci,
        capsize=5,
        color="#e67e73",
        edgecolor="#c0392b",
        label="Mean Risk",
    )
    ax.bar(
        x + width / 2,
        safety_means,
        width,
        yerr=safety_ci,
        capsize=5,
        color="#7fb3d5",
        edgecolor="#1f77b4",
        label="Mean Safety",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_title("Mean Risk and Safety Score by Model")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_summary(stats: List[ModelStats], output_path: Path) -> None:
    lines = []
    lines.append("model,metric,mean_from_md,mean_from_csv,ci95,n")
    for row in stats:
        lines.append(
            f"{row.model},Risk,{row.risk.mean_from_md:.6f},{row.risk.mean_from_csv:.6f},{row.risk.ci95:.6f},{row.risk.n}"
        )
        lines.append(
            f"{row.model},Safety,{row.safety.mean_from_md:.6f},{row.safety.mean_from_csv:.6f},{row.safety.ci95:.6f},{row.safety.n}"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["auto_attack", "reframe"],
        default="auto_attack",
        help="Result mode to use for loading per-row judge CSV files and output names.",
    )
    parser.add_argument(
        "--gemini-report",
        type=Path,
        default=Path("outputs/safety_exp/results/auto_attack/gemini-3-pro/summary/auto_attack_codebook_report.md"),
    )
    parser.add_argument(
        "--gpt4o-report",
        type=Path,
        default=Path("outputs/safety_exp/results/auto_attack/gpt4o/summary/auto_attack_codebook_report.md"),
    )
    parser.add_argument(
        "--claude-report",
        type=Path,
        default=Path("outputs/safety_exp/results/auto_attack/claude-4.5-opus/summary/auto_attack_codebook_report.md"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/safety_exp/fig"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_inputs = [
        ("Gemini-3-pro", args.gemini_report),
        ("Gpt-4o", args.gpt4o_report),
        ("Claude-4.5-opus", args.claude_report),
    ]
    stats = [build_stats(model_name, report_path, mode=args.mode) for model_name, report_path in model_inputs]

    draw_dot_plot(stats, args.out_dir / f"{args.mode}_risk_safety_dot_ci.png", mode=args.mode)
    draw_bar_plot(stats, args.out_dir / f"{args.mode}_risk_safety_bar_ci.png", mode=args.mode)
    write_summary(stats, args.out_dir / f"{args.mode}_risk_safety_ci_summary.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
