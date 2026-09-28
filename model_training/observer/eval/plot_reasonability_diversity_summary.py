#!/usr/bin/env python3
"""Plot reasonability and diversity tradeoffs for auto evaluation runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from angel_common.paths import OUTPUTS_DIR


MODEL_LABELS = {
    "claude": "Claude",
    "gemini": "Gemini",
    "qwen3_8b": "Ours",
    "ours": "Qwen3-8B",
}

MODEL_COLORS = {
    "Claude": "#1f77b4",
    "Gemini": "#ff7f0e",
    "Ours": "#d62728",
    "Qwen3-8B": "#2ca02c",
}

EXCLUDED_MODELS = {"gpt5"}


def read_diversity_at_k(summary_path: Path, k: int) -> float:
    with summary_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row["num_profiles_per_short_profile"]) == k:
                return float(row["avg_mean_pairwise_cosine_distance"])
    raise ValueError(f"No diversity row for k={k} in {summary_path}")


def load_rows(runs_dir: Path, k: int) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for report_path in sorted(runs_dir.glob("*/report.json")):
        model_key = report_path.parent.name
        if model_key in EXCLUDED_MODELS or model_key not in MODEL_LABELS:
            continue

        diversity_path = report_path.parent / "diversity" / "diversity_summary_by_k.jsonl"
        if not diversity_path.exists():
            continue

        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)

        rows.append(
            {
                "model": MODEL_LABELS[model_key],
                "reasonability": float(report["reasonability"]["avg_reasonability_score"]),
                "edges": float(report["reasonability"]["avg_edges_per_profile"]),
                "diversity": read_diversity_at_k(diversity_path, k),
            }
        )
    return rows


def pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    return numerator / (denom_x * denom_y)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#d9dee7", linewidth=0.8)
    ax.grid(True, axis="x", color="#eef1f5", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate(ax: plt.Axes, x: float, y: float, label: str, *, context: str) -> None:
    offsets = {
        ("tradeoff", "Ours"): (10, 8),
        ("tradeoff", "Claude"): (10, -2),
        ("tradeoff", "Gemini"): (10, 8),
        ("edges", "Ours"): (8, 6),
        ("edges", "Claude"): (8, 16),
        ("edges", "Gemini"): (8, 4),
    }
    dx, dy = offsets.get((context, label), (6, 5))
    ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9)


def add_trend_line(ax: plt.Axes, xs: list[float], ys: list[float]) -> None:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sum(
        (x - mean_x) ** 2 for x in xs
    )
    intercept = mean_y - slope * mean_x
    line_xs = [min(xs), max(xs)]
    line_ys = [slope * x + intercept for x in line_xs]
    ax.plot(line_xs, line_ys, color="#6b7280", linewidth=1.4, linestyle="--", zorder=1)


def plot_tradeoff_score(rows: list[dict[str, float | str]], output_prefix: Path, k: int) -> None:
    labels = [str(row["model"]) for row in rows]
    reasonability = [float(row["reasonability"]) for row in rows]
    diversity = [float(row["diversity"]) for row in rows]
    max_reasonability = max(reasonability)
    max_diversity = max(diversity)

    reasonability_retention = [score / max_reasonability for score in reasonability]
    diversity_retention = [
        max(0.0, score / max_diversity) if max_diversity else 0.0 for score in diversity
    ]
    tradeoff_score = [
        math.sqrt(r_score * d_score)
        for r_score, d_score in zip(reasonability_retention, diversity_retention)
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    colors = [MODEL_COLORS[label] for label in labels]
    bars = ax.bar(labels, tradeoff_score, color=colors, width=0.64)
    ax.set_title(f"Reasonability-Diversity Tradeoff Score (k={k})")
    ax.set_ylabel("sqrt(reasonability retention x diversity retention)")
    ax.set_ylim(0, 1.08)
    style_axis(ax)
    for bar, value in zip(bars, tradeoff_score):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(output_prefix.with_name("tradeoff_score_no_gpt5.png"), bbox_inches="tight")
    fig.savefig(output_prefix.with_name("tradeoff_score_no_gpt5.pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=OUTPUTS_DIR / "observer" / "auto_eval_runs")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=OUTPUTS_DIR / "observer" / "auto_eval_runs" / "reasonability_diversity",
    )
    args = parser.parse_args()

    rows = load_rows(args.runs_dir, args.k)
    if not rows:
        raise SystemExit(f"No comparable rows found under {args.runs_dir}")

    rows = sorted(rows, key=lambda row: str(row["model"]))
    labels = [str(row["model"]) for row in rows]
    colors = [MODEL_COLORS[label] for label in labels]

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=200)
    values = [float(row["reasonability"]) for row in rows]
    bars = ax.bar(labels, values, color=colors, width=0.64)
    ax.set_title("Reasonability by Model")
    ax.set_ylabel("Avg. reasonability score")
    ax.set_ylim(0.84, 0.92)
    style_axis(ax)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.002,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(args.output_prefix.with_name("reasonability_no_gpt5.png"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_name("reasonability_no_gpt5.pdf"), bbox_inches="tight")
    plt.close(fig)

    plot_tradeoff_score(rows, args.output_prefix, args.k)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    xs = [float(row["edges"]) for row in rows]
    ys = [float(row["reasonability"]) for row in rows]
    r = pearson(xs, ys)
    add_trend_line(ax, xs, ys)
    for row in rows:
        label = str(row["model"])
        x = float(row["edges"])
        y = float(row["reasonability"])
        ax.scatter(x, y, s=90, color=MODEL_COLORS[label], zorder=2)
        annotate(ax, x, y, label, context="reasonability_edges")
    ax.set_title(f"Reasonability vs. Avg. Edges (Pearson r={r:.3f}, n={len(rows)})")
    ax.set_xlabel("Avg. edges per profile")
    ax.set_ylabel("Avg. reasonability score")
    ax.margins(x=0.08, y=0.12)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(args.output_prefix.with_name("reasonability_vs_edges_no_gpt5.png"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_name("reasonability_vs_edges_no_gpt5.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    for row in rows:
        label = str(row["model"])
        x = float(row["diversity"])
        y = float(row["reasonability"])
        ax.scatter(x, y, s=90, color=MODEL_COLORS[label])
        annotate(ax, x, y, label, context="tradeoff")
    ax.set_title(f"Reasonability-Diversity Tradeoff (k={args.k})")
    ax.set_xlabel("Avg. mean pairwise cosine distance")
    ax.set_ylabel("Avg. reasonability score")
    ax.set_ylim(0, 1.0)
    ax.margins(x=0.08, y=0.12)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(args.output_prefix.with_name("reasonability_vs_diversity_no_gpt5.png"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_name("reasonability_vs_diversity_no_gpt5.pdf"), bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
    xs = [float(row["edges"]) for row in rows]
    ys = [float(row["diversity"]) for row in rows]
    r = pearson(xs, ys)
    add_trend_line(ax, xs, ys)
    for row in rows:
        label = str(row["model"])
        x = float(row["edges"])
        y = float(row["diversity"])
        ax.scatter(x, y, s=90, color=MODEL_COLORS[label], zorder=2)
        annotate(ax, x, y, label, context="edges")
    ax.set_title(f"Avg. Edges vs. Diversity (k={args.k}, Pearson r={r:.3f}, n={len(rows)})")
    ax.set_xlabel("Avg. edges per profile")
    ax.set_ylabel("Avg. mean pairwise cosine distance")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.08, y=0.12)
    style_axis(ax)
    fig.tight_layout()
    fig.savefig(args.output_prefix.with_name("edges_vs_diversity_no_gpt5.png"), bbox_inches="tight")
    fig.savefig(args.output_prefix.with_name("edges_vs_diversity_no_gpt5.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
