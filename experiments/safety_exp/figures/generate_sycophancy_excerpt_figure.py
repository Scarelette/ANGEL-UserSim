#!/usr/bin/env python3
"""Generate a paper-style excerpt figure for one codebook pattern pair."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path(
            "outputs/safety_exp/results/auto_attack/gpt4o/summary/"
            "auto_attack_codebook_pattern_top3_risky_pairs_presentation.jsonl"
        ),
    )
    parser.add_argument("--pattern", default="Sycophancy")
    parser.add_argument(
        "--pair-rank",
        type=int,
        default=1,
        help="1-based rank inside top3_prompt_response_pairs.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("outputs/safety_exp/fig/fig4/sycophancy_excerpt_example.png"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("outputs/safety_exp/fig/fig4/sycophancy_excerpt_example.pdf"),
    )
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def load_pattern_pair(input_jsonl: Path, pattern_name: str, pair_rank: int) -> Tuple[Dict, Dict]:
    if pair_rank < 1:
        raise ValueError("pair-rank must be >= 1")
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")

    pattern_obj = None
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("codebook_pattern_name") == pattern_name:
                pattern_obj = obj
                break
    if pattern_obj is None:
        raise ValueError(f"Pattern '{pattern_name}' not found in {input_jsonl}")

    pairs = pattern_obj.get("top3_prompt_response_pairs", [])
    if not pairs:
        raise ValueError(f"No pairs found for pattern '{pattern_name}'")
    if pair_rank > len(pairs):
        raise ValueError(f"pair-rank {pair_rank} out of range; available: 1..{len(pairs)}")
    pair = pairs[pair_rank - 1]
    return pattern_obj, pair


def parse_excerpt_lines(excerpt: str, wrap_width: int) -> List[Tuple[str, str]]:
    """Return list of (line, kind) where kind in {'normal','highlight','omit'}."""
    out: List[Tuple[str, str]] = []
    for raw in str(excerpt or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "[ ... ]":
            out.append((line, "omit"))
            continue
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            core = line[2:-2].strip()
            wrapped = textwrap.wrap(core, width=wrap_width) or [core]
            out.extend((w, "highlight") for w in wrapped)
            continue
        wrapped = textwrap.wrap(line, width=wrap_width) or [line]
        out.extend((w, "normal") for w in wrapped)
    return out


def draw_text_panel(
    ax: plt.Axes,
    title: str,
    lines: List[Tuple[str, str]],
    panel_face: str,
    panel_edge: str,
) -> None:
    ax.set_axis_off()
    # Outer rounded panel.
    panel = patches.FancyBboxPatch(
        (0.015, 0.02),
        0.97,
        0.96,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=panel_edge,
        facecolor=panel_face,
        transform=ax.transAxes,
    )
    ax.add_patch(panel)

    ax.text(
        0.04,
        0.955,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )

    # Dynamic spacing based on line count.
    n_lines = max(len(lines), 1)
    usable_h = 0.86
    line_h = min(0.038, usable_h / (n_lines + 1))
    y = 0.90

    for text, kind in lines:
        if y < 0.05:
            ax.text(
                0.05,
                0.045,
                "...",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=10,
                color="#64748b",
            )
            break
        if kind == "omit":
            ax.text(
                0.05,
                y,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10.5,
                color="#64748b",
                style="italic",
            )
        elif kind == "highlight":
            ax.text(
                0.05,
                y,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10.5,
                fontweight="bold",
                color="#7f1d1d",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="#fee2e2",
                    edgecolor="#ef4444",
                    linewidth=0.8,
                ),
            )
        else:
            ax.text(
                0.05,
                y,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10.5,
                color="#0f172a",
            )
        y -= line_h


def main() -> int:
    args = parse_args()
    pattern_obj, pair = load_pattern_pair(args.input_jsonl, args.pattern, args.pair_rank)

    prompt_lines = parse_excerpt_lines(pair.get("prompt_excerpt", ""), wrap_width=62)
    response_lines = parse_excerpt_lines(pair.get("response_excerpt", ""), wrap_width=62)

    reason_text = str(pair.get("risky_reason_focus") or "").strip()
    reason_wrap = textwrap.fill(reason_text, width=150)

    fig = plt.figure(figsize=(15.8, 9.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.18, 0.66, 0.16], hspace=0.14, wspace=0.08)

    # Header
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_axis_off()
    ax_h.text(
        0.5,
        0.80,
        f"Example Excerpt Figure: {pattern_obj.get('codebook_pattern_name', args.pattern)} Pattern",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#0f172a",
    )
    meta = (
        f"Pair {pair.get('rank')} | Prompt key: {pair.get('prompt_key')} | "
        f"Pattern score: {pair.get('pattern_score')} | Risk: {pair.get('risk')} | Safety: {pair.get('safety')}"
    )
    ax_h.text(0.5, 0.47, meta, ha="center", va="center", fontsize=12, color="#334155")
    ax_h.text(
        0.5,
        0.21,
        "Red highlighted lines indicate risky signals in the response excerpt.",
        ha="center",
        va="center",
        fontsize=11,
        color="#64748b",
    )

    # Prompt / Response panels
    ax_l = fig.add_subplot(gs[1, 0])
    draw_text_panel(
        ax_l,
        "Prompt Excerpt",
        prompt_lines,
        panel_face="#fff9ec",
        panel_edge="#f59e0b",
    )

    ax_r = fig.add_subplot(gs[1, 1])
    draw_text_panel(
        ax_r,
        "Response Excerpt (Risky Signals Highlighted)",
        response_lines,
        panel_face="#eff8ff",
        panel_edge="#38bdf8",
    )

    # Reason panel
    ax_b = fig.add_subplot(gs[2, :])
    ax_b.set_axis_off()
    reason_box = patches.FancyBboxPatch(
        (0.01, 0.08),
        0.98,
        0.84,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.3,
        edgecolor="#94a3b8",
        facecolor="#f8fafc",
        transform=ax_b.transAxes,
    )
    ax_b.add_patch(reason_box)
    ax_b.text(
        0.03,
        0.82,
        "Judge Risky Reason Focus",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
    )
    ax_b.text(
        0.03,
        0.63,
        reason_wrap,
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        color="#334155",
    )

    for path in [args.output_png, args.output_pdf]:
        path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.output_pdf, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(args.output_png)
    print(args.output_pdf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

