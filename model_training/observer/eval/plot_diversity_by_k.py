#!/usr/bin/env python3
"""Plot diversity curves from auto evaluation run summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from angel_common.paths import OUTPUTS_DIR


MODEL_LABELS = {
    "claude": "Claude",
    "gemini": "Gemini",
    "qwen3_8b": "Ours",
    "ours": "Qwen3-8B",
}

EXCLUDED_MODELS = {"gpt5"}


def read_series(path: Path) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            xs.append(int(row["num_profiles_per_short_profile"]))
            ys.append(float(row["avg_mean_pairwise_cosine_distance"]))

    pairs = sorted(zip(xs, ys))
    return [x for x, _ in pairs], [y for _, y in pairs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=OUTPUTS_DIR / "observer" / "auto_eval_runs",
        help="Directory containing one subdirectory per model run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "observer" / "auto_eval_runs" / "diversity_avg_mean_pairwise_cosine_distance.png",
        help="Output figure path. A PDF copy is also written next to it.",
    )
    args = parser.parse_args()

    run_paths = sorted(args.runs_dir.glob("*/diversity/diversity_summary_by_k.jsonl"))
    if not run_paths:
        raise SystemExit(f"No diversity summaries found under {args.runs_dir}")

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)

    for summary_path in run_paths:
        model_key = summary_path.parents[1].name
        if model_key in EXCLUDED_MODELS:
            continue
        label = MODEL_LABELS.get(model_key, model_key)
        xs, ys = read_series(summary_path)
        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=4.5, label=label)

    ax.set_title("Diversity by Number of Generated Profiles")
    ax.set_xlabel("Number of profiles per short profile")
    ax.set_ylabel("Avg. mean pairwise cosine distance")
    ax.set_xticks(range(2, 13))
    ax.set_ylim(bottom=0)
    ax.grid(True, axis="y", color="#d9dee7", linewidth=0.8)
    ax.grid(True, axis="x", color="#eef1f5", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="Model", frameon=False, ncols=2)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
