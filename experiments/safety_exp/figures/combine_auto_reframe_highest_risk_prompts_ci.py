#!/usr/bin/env python3
"""Combine auto-generation vs reframe Highest-risk Prompts plots per model (bar/dot, 95% CI)."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Match requested yellow/blue style.
AUTO_FILL = "#ffc37d"
AUTO_EDGE = "#d9861e"
REF_FILL = "#56b4e9"
REF_EDGE = "#1f77b4"
TEXT_DARK = "#1f2937"


SHORT_PROMPT_LABELS: Dict[str, str] = {
    "grandiosity": "grand.",
    "termination": "term.",
    "consciousness": "consc.",
    "experimentation": "exper.",
    "morality": "moral.",
    "withdrawal": "withdraw.",
    "medication": "med.",
    "reality test": "reality",
    "safety test": "safety",
    "solipsism": "solips.",
    "concealment": "conceal.",
    "romance": "romance",
    "letter": "letter",
    "thought disorder": "thought dis.",
    "thought insertion": "thought ins.",
    "bizarre delusion": "bizarre del.",
}


def pretty_model_name(model_dir: str) -> str:
    if model_dir == "gpt4o":
        return "gpt-4o"
    return model_dir


def slug(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "-" for ch in text.lower()).strip("-")


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def std_sample(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def ci95(values: List[float]) -> float:
    if not values:
        return 0.0
    return 1.96 * std_sample(values) / math.sqrt(len(values))


def short_label(prompt: str) -> str:
    if prompt in SHORT_PROMPT_LABELS:
        return SHORT_PROMPT_LABELS[prompt]
    if len(prompt) <= 10:
        return prompt
    return prompt[:9] + "."


def parse_highest_risk_table(report_path: Path) -> List[Tuple[str, float]]:
    lines = report_path.read_text(encoding="utf-8").splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if line.strip() == "## Highest-risk Prompts (Mean Risk)":
            start_idx = idx
            break
    if start_idx is None:
        raise ValueError(f"Could not find 'Highest-risk Prompts (Mean Risk)' in {report_path}")

    rows: List[Tuple[str, float]] = []
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if not stripped.startswith("|"):
            continue
        if "Rank" in stripped and "Prompt" in stripped:
            continue
        if set(stripped.replace("|", "").strip()) <= {"-", ":"}:
            continue
        parts = [p.strip() for p in stripped.split("|")[1:-1]]
        if len(parts) < 5:
            continue
        prompt = parts[1]
        mean_risk = float(parts[3])
        rows.append((prompt, mean_risk))

    if not rows:
        raise ValueError(f"No rows parsed from Highest-risk Prompts table in {report_path}")
    return rows


def list_judge_files(result_dir: Path, mode: str) -> List[Path]:
    if mode == "auto_attack":
        patterns = [
            "codebook_judge_results_full_auto_attack_profile*.jsonl",
            "codebook_judge_results_full_profile*.jsonl",
        ]
    else:
        patterns = [
            "codebook_judge_results_full_reframe_profile*.jsonl",
            "codebook_judge_results_full_profile*.jsonl",
        ]

    for pattern in patterns:
        files = sorted(result_dir.glob(pattern))
        if files:
            return files
    return []


def load_prompt_risk_values(result_dir: Path, mode: str) -> Dict[str, List[float]]:
    by_prompt: Dict[str, List[float]] = defaultdict(list)
    judge_files = list_judge_files(result_dir, mode)
    if not judge_files:
        raise ValueError(f"No judge files found in {result_dir} for mode={mode}")

    for judge_file in judge_files:
        with judge_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("error") is not None:
                    continue
                judgment = payload.get("judgment")
                csv_row = payload.get("csv_row")
                if not isinstance(judgment, dict) or not isinstance(csv_row, dict):
                    continue

                prompt = str(payload.get("prompt_raw") or payload.get("prompt") or "").strip()
                if not prompt:
                    continue
                try:
                    risk_val = float(csv_row.get("Risk", 0.0))
                except (TypeError, ValueError):
                    continue
                by_prompt[prompt].append(risk_val)

    return by_prompt


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
    auto_prompts: List[str],
    auto_means: List[float],
    auto_cis: List[float],
    ref_prompts: List[str],
    ref_means: List[float],
    ref_cis: List[float],
    out_path: Path,
    dpi: int,
) -> None:
    n_left = len(auto_prompts)
    n_right = len(ref_prompts)
    x_left = np.arange(n_left, dtype=float)
    gap = 1.1
    x_right = np.arange(n_right, dtype=float) + n_left + gap
    divider_x = n_left - 0.5 + gap / 2

    ymax = max(
        max(m + c for m, c in zip(auto_means, auto_cis)),
        max(m + c for m, c in zip(ref_means, ref_cis)),
    )
    ylim_top = ymax * 1.13 if ymax > 0 else 1.0

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 34,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 16,
        }
    )

    fig, ax = plt.subplots(figsize=(25, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfdff")

    bars_left = ax.bar(
        x_left,
        auto_means,
        width=0.78,
        yerr=auto_cis,
        capsize=4,
        color=AUTO_FILL,
        edgecolor=AUTO_EDGE,
        linewidth=1.5,
        alpha=0.95,
    )
    bars_right = ax.bar(
        x_right,
        ref_means,
        width=0.78,
        yerr=ref_cis,
        capsize=4,
        color=REF_FILL,
        edgecolor=REF_EDGE,
        linewidth=1.5,
        alpha=0.95,
    )

    annotate(
        ax,
        np.array([b.get_x() + b.get_width() / 2 for b in bars_left]),
        auto_means,
        auto_cis,
        ylim_top,
    )
    annotate(
        ax,
        np.array([b.get_x() + b.get_width() / 2 for b in bars_right]),
        ref_means,
        ref_cis,
        ylim_top,
    )

    xlabels = [short_label(p) for p in auto_prompts] + [short_label(p) for p in ref_prompts]
    xticks = np.concatenate([x_left, x_right])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=22, ha="right")

    ax.axvspan(-0.8, n_left - 0.3, color="#fff9ec", zorder=0, alpha=0.7)
    ax.axvspan(n_left + gap - 0.7, n_left + gap + n_right - 0.25, color="#eff8ff", zorder=0, alpha=0.7)
    ax.axvline(divider_x, color="#64748b", linestyle=(0, (4, 3)), linewidth=1.5, alpha=0.9)

    ax.text((x_left[0] + x_left[-1]) / 2, ylim_top * 0.985, "Auto-generation", ha="center", va="top", fontsize=24, fontweight="bold", color=TEXT_DARK)
    ax.text((x_right[0] + x_right[-1]) / 2, ylim_top * 0.985, "Reframe", ha="center", va="top", fontsize=24, fontweight="bold", color=TEXT_DARK)

    ax.set_xlim(-0.9, x_right[-1] + 0.9)
    ax.set_ylim(0, ylim_top)
    ax.set_ylabel("Score")
    ax.set_xlabel("Prompt")
    ax.set_title(f"Highest-risk Prompts ({model_name})", pad=16, fontweight="bold")
    ax.grid(axis="y", alpha=0.22, color="#94a3b8")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.75)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def draw_combined_dot(
    model_name: str,
    auto_prompts: List[str],
    auto_means: List[float],
    auto_cis: List[float],
    ref_prompts: List[str],
    ref_means: List[float],
    ref_cis: List[float],
    out_path: Path,
    dpi: int,
) -> None:
    n_left = len(auto_prompts)
    n_right = len(ref_prompts)
    x_left = np.arange(n_left, dtype=float)
    gap = 1.1
    x_right = np.arange(n_right, dtype=float) + n_left + gap
    divider_x = n_left - 0.5 + gap / 2

    ymax = max(
        max(m + c for m, c in zip(auto_means, auto_cis)),
        max(m + c for m, c in zip(ref_means, ref_cis)),
    )
    ylim_top = ymax * 1.13 if ymax > 0 else 1.0

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.titlesize": 34,
            "axes.labelsize": 20,
            "xtick.labelsize": 15,
            "ytick.labelsize": 16,
        }
    )

    fig, ax = plt.subplots(figsize=(25, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfdff")

    ax.errorbar(
        x_left,
        auto_means,
        yerr=auto_cis,
        fmt="o",
        markersize=9,
        capsize=4,
        linewidth=1.9,
        color=AUTO_EDGE,
        markerfacecolor=AUTO_FILL,
        markeredgewidth=1.5,
    )
    ax.errorbar(
        x_right,
        ref_means,
        yerr=ref_cis,
        fmt="o",
        markersize=9,
        capsize=4,
        linewidth=1.9,
        color=REF_EDGE,
        markerfacecolor=REF_FILL,
        markeredgewidth=1.5,
    )

    annotate(ax, x_left, auto_means, auto_cis, ylim_top)
    annotate(ax, x_right, ref_means, ref_cis, ylim_top)

    xlabels = [short_label(p) for p in auto_prompts] + [short_label(p) for p in ref_prompts]
    xticks = np.concatenate([x_left, x_right])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, rotation=22, ha="right")

    ax.axvspan(-0.8, n_left - 0.3, color="#fff9ec", zorder=0, alpha=0.7)
    ax.axvspan(n_left + gap - 0.7, n_left + gap + n_right - 0.25, color="#eff8ff", zorder=0, alpha=0.7)
    ax.axvline(divider_x, color="#64748b", linestyle=(0, (4, 3)), linewidth=1.5, alpha=0.9)

    ax.text((x_left[0] + x_left[-1]) / 2, ylim_top * 0.985, "Auto-generation", ha="center", va="top", fontsize=24, fontweight="bold", color=TEXT_DARK)
    ax.text((x_right[0] + x_right[-1]) / 2, ylim_top * 0.985, "Reframe", ha="center", va="top", fontsize=24, fontweight="bold", color=TEXT_DARK)

    ax.set_xlim(-0.9, x_right[-1] + 0.9)
    ax.set_ylim(0, ylim_top)
    ax.set_ylabel("Score")
    ax.set_xlabel("Prompt")
    ax.set_title(f"Highest-risk Prompts ({model_name})", pad=16, fontweight="bold")
    ax.grid(axis="y", alpha=0.22, color="#94a3b8")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.75)
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
        default=Path("outputs/safety_exp/fig/fig3"),
    )
    parser.add_argument(
        "--chart-type",
        choices=["bar", "dot", "both"],
        default="both",
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
    ref_reports = sorted(args.reframe_root.glob("*/summary/reframe_codebook_report.md"))
    auto_by_model = {p.parent.parent.name: p for p in auto_reports}
    ref_by_model = {p.parent.parent.name: p for p in ref_reports}
    common_models = sorted(set(auto_by_model) & set(ref_by_model))
    if not common_models:
        raise SystemExit("No overlapping models found between auto_attack and reframe reports.")

    for model in common_models:
        auto_report = auto_by_model[model]
        ref_report = ref_by_model[model]
        auto_dir = auto_report.parent.parent
        ref_dir = ref_report.parent.parent

        auto_table = parse_highest_risk_table(auto_report)
        ref_table = parse_highest_risk_table(ref_report)
        auto_vals = load_prompt_risk_values(auto_dir, "auto_attack")
        ref_vals = load_prompt_risk_values(ref_dir, "reframe")

        auto_prompts: List[str] = []
        auto_means: List[float] = []
        auto_cis: List[float] = []
        for prompt, m in auto_table:
            vals = auto_vals.get(prompt, [])
            if not vals:
                continue
            auto_prompts.append(prompt)
            auto_means.append(m)
            auto_cis.append(ci95(vals))

        ref_prompts: List[str] = []
        ref_means: List[float] = []
        ref_cis: List[float] = []
        for prompt, m in ref_table:
            vals = ref_vals.get(prompt, [])
            if not vals:
                continue
            ref_prompts.append(prompt)
            ref_means.append(m)
            ref_cis.append(ci95(vals))

        if not auto_prompts or not ref_prompts:
            print(f"Skipping {model}: missing prompts in one mode.")
            continue

        model_pretty = pretty_model_name(model)

        if args.chart_type in ("bar", "both"):
            out_path = args.out_dir / f"{slug(model)}_auto_vs_reframe_highest_risk_prompts_bar_ci.png"
            draw_combined_bar(
                model_pretty,
                auto_prompts,
                auto_means,
                auto_cis,
                ref_prompts,
                ref_means,
                ref_cis,
                out_path,
                args.dpi,
            )
            print(out_path)

        if args.chart_type in ("dot", "both"):
            out_path = args.out_dir / f"{slug(model)}_auto_vs_reframe_highest_risk_prompts_dot_ci.png"
            draw_combined_dot(
                model_pretty,
                auto_prompts,
                auto_means,
                auto_cis,
                ref_prompts,
                ref_means,
                ref_cis,
                out_path,
                args.dpi,
            )
            print(out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
