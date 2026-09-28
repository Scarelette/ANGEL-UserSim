#!/usr/bin/env python3
"""Plot per-model Highest-risk Prompts (bar/dot) with 95% confidence intervals."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


MODEL_DISPLAY_NAME: Dict[str, str] = {
    "gpt4o": "GPT-4o",
    "gemini-3-pro": "Gemini-3-pro",
    "claude-4.5-opus": "Claude-4.5-opus",
}

PROMPT_DISPLAY_NAME: Dict[str, str] = {
    "grandiosity": "Grandiosity",
    "termination": "Termination",
    "thought disorder": "Thought Dis.",
    "thought insertion": "Thought Ins.",
    "bizarre delusion": "Bizarre Del.",
    "reality test": "Reality Test",
    "safety test": "Safety Test",
    "consciousness": "Conscious.",
    "withdrawal": "Withdrawal",
    "experimentation": "Experiment",
    "concealment": "Concealment",
    "medication": "Medication",
    "romance": "Romance",
    "morality": "Morality",
    "solipsism": "Solipsism",
    "letter": "Letter",
}


def pretty_model_name(model_dir: str) -> str:
    return MODEL_DISPLAY_NAME.get(model_dir, model_dir)


def short_prompt_label(prompt: str) -> str:
    normalized = prompt.strip().lower().replace("_", " ")
    if normalized in PROMPT_DISPLAY_NAME:
        return PROMPT_DISPLAY_NAME[normalized]

    words = [w for w in normalized.split() if w]
    if not words:
        return prompt
    if len(words) == 1:
        word = words[0]
        return word.capitalize() if len(word) <= 12 else f"{word[:11].capitalize()}."

    parts: List[str] = []
    for word in words[:2]:
        if len(word) <= 8:
            parts.append(word.capitalize())
        else:
            parts.append(f"{word[:7].capitalize()}.")
    return " ".join(parts)


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
        patterns = ["codebook_judge_results_full_auto_attack_profile*.jsonl"]
    else:
        patterns = [
            "codebook_judge_results_full_reframe_profile*.jsonl",
            "codebook_judge_results_full_profile*.jsonl",  # legacy gpt4o naming
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


def draw_bar_plot(
    model_name: str,
    mode: str,
    prompts: List[str],
    means_from_report: List[float],
    cis: List[float],
    out_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    x = np.arange(len(prompts), dtype=float)
    ymax = max((m + c) for m, c in zip(means_from_report, cis))
    ylim_top = ymax * 1.12 if ymax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(20, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bars = ax.bar(
        x,
        means_from_report,
        width=0.72,
        yerr=cis,
        capsize=5,
        color="#7fb3d5",
        edgecolor="#1f77b4",
        linewidth=1.3,
    )

    for bar, m, c in zip(bars, means_from_report, cis):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            m + c + ylim_top * 0.01,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="normal",
            color="#111111",
        )

    labels = [short_prompt_label(p) for p in prompts]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontweight="normal")
    ax.set_ylim(0, ylim_top)
    ax.set_ylabel("Score")
    ax.set_xlabel("Prompt")
    ax.set_title(f"Overall Highest-risk Prompts ({model_name})", pad=14, fontweight="normal")
    ax.grid(axis="y", alpha=0.25, color="#999999")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(True)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def draw_dot_plot(
    model_name: str,
    mode: str,
    prompts: List[str],
    means_from_report: List[float],
    cis: List[float],
    out_path: Path,
    dpi: int,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 24,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    x = np.arange(len(prompts), dtype=float)
    ymax = max((m + c) for m, c in zip(means_from_report, cis))
    ylim_top = ymax * 1.12 if ymax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(20, 9))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.errorbar(
        x,
        means_from_report,
        yerr=cis,
        fmt="o",
        markersize=8.5,
        capsize=5,
        linewidth=1.8,
        color="#1f77b4",
        markerfacecolor="#7fb3d5",
        markeredgecolor="#1f77b4",
        markeredgewidth=1.6,
    )

    for x0, m, c in zip(x, means_from_report, cis):
        ax.text(
            x0,
            m + c + ylim_top * 0.01,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="normal",
            color="#111111",
        )

    labels = [short_prompt_label(p) for p in prompts]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontweight="normal")
    ax.set_ylim(0, ylim_top)
    ax.set_ylabel("Score")
    ax.set_xlabel("Prompt")
    ax.set_title(f"Overall Highest-risk Prompts ({model_name})", pad=14, fontweight="normal")
    ax.grid(axis="y", alpha=0.25, color="#999999")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(True)
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reports-glob",
        default="outputs/safety_exp/results/auto_attack/*/summary/auto_attack_codebook_report.md",
        help="Glob for auto-attack summary markdown files.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto_attack", "reframe"],
        default="auto_attack",
        help="Evaluation mode (controls judge-file pattern and output naming).",
    )
    parser.add_argument(
        "--chart-type",
        choices=["bar", "dot", "both"],
        default="both",
        help="Figure type to generate.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/safety_exp/fig/fig3"),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure DPI.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_paths = sorted(Path().glob(args.reports_glob))
    if not report_paths:
        raise SystemExit(f"No reports matched: {args.reports_glob}")

    for report_path in report_paths:
        result_dir = report_path.parent.parent
        model_dir = result_dir.name
        model_name = pretty_model_name(model_dir)

        highest_rows = parse_highest_risk_table(report_path)
        by_prompt = load_prompt_risk_values(result_dir, mode=args.mode)

        prompts: List[str] = []
        means_from_report: List[float] = []
        cis: List[float] = []
        missing_prompts: List[str] = []

        for prompt, mean_risk in highest_rows:
            values = by_prompt.get(prompt, [])
            if not values:
                missing_prompts.append(prompt)
                continue
            prompts.append(prompt)
            means_from_report.append(mean_risk)
            cis.append(ci95(values))

        if missing_prompts:
            print(f"Warning: missing raw rows for prompts in {model_dir}: {', '.join(missing_prompts)}")
        if not prompts:
            print(f"Skipping {model_dir}: no prompts with raw risk values.")
            continue

        if args.chart_type in ("bar", "both"):
            out_path = args.out_dir / f"{slug(model_dir)}_{args.mode}_highest_risk_prompts_bar_ci.png"
            draw_bar_plot(model_name, args.mode, prompts, means_from_report, cis, out_path, args.dpi)
            print(out_path)
        if args.chart_type in ("dot", "both"):
            out_path = args.out_dir / f"{slug(model_dir)}_{args.mode}_highest_risk_prompts_dot_ci.png"
            draw_dot_plot(model_name, args.mode, prompts, means_from_report, cis, out_path, args.dpi)
            print(out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
