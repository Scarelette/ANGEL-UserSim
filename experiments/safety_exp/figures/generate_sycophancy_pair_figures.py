#!/usr/bin/env python3
"""Generate one case figure per codebook prompt-response pair."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches


def redact_lee(text: str) -> str:
    """Remove personal name 'Lee' from rendered text content."""
    t = str(text or "")
    t = re.sub(r"\b[Ll]ee['’]s\b", "the character's", t)
    t = re.sub(r"\b[Ll]ee\b", "the character", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path(
            "outputs/safety_exp/results/auto_attack/gpt4o/summary/"
            "auto_attack_codebook_pattern_top6_risky_pairs_presentation.jsonl"
        ),
    )
    parser.add_argument("--pattern", default="Sycophancy")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/safety_exp/fig/fig4_auto_attack/pattern_pair_cases"
        ),
    )
    parser.add_argument("--max-pairs", type=int, default=0, help="0 means all pairs.")
    parser.add_argument(
        "--palette",
        default="rose_teal",
        choices=["rose_teal", "peach_mint", "butter_sky", "cream_indigo"],
    )
    parser.add_argument("--max-prompt-sentences", type=int, default=8)
    parser.add_argument("--max-response-sentences", type=int, default=11)
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def get_palette(name: str) -> Dict[str, str]:
    palettes = {
        "peach_mint": {
            "prompt_face": "#fff4e8",
            "prompt_edge": "#f59e0b",
            "resp_face": "#ecfdf5",
            "resp_edge": "#10b981",
            "fig_face": "#f7fbff",
        },
        "rose_teal": {
            "prompt_face": "#FFECEF",
            "prompt_edge": "#EB5757",
            "resp_face": "#E8F9F5",
            "resp_edge": "#1ABC9C",
            "fig_face": "#fbfdff",
        },
        "butter_sky": {
            "prompt_face": "#FFF4CC",
            "prompt_edge": "#E6A700",
            "resp_face": "#EAF4FF",
            "resp_edge": "#2F80ED",
            "fig_face": "#fbfdff",
        },
        "cream_indigo": {
            "prompt_face": "#FFF8E1",
            "prompt_edge": "#D4A017",
            "resp_face": "#EEF2FF",
            "resp_edge": "#4F46E5",
            "fig_face": "#fbfdff",
        },
    }
    return palettes[name]


def load_pattern_obj(input_jsonl: Path, pattern: str) -> Dict:
    if not input_jsonl.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_jsonl}")
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("codebook_pattern_name") == pattern:
                return obj
    raise ValueError(f"Pattern '{pattern}' not found in {input_jsonl}")


def get_ranked_pairs(pattern_obj: Dict) -> List[Dict]:
    for key in (
        "top10_prompt_response_pairs",
        "top6_prompt_response_pairs",
        "top3_prompt_response_pairs",
    ):
        pairs = pattern_obj.get(key, [])
        if isinstance(pairs, list) and pairs:
            return list(pairs)
    return []


def parse_excerpt_lines(excerpt: str) -> List[Tuple[str, str]]:
    """Return (line, kind) where kind in {'normal','highlight','omit'}."""
    lines: List[Tuple[str, str]] = []
    for raw in str(excerpt or "").splitlines():
        line = redact_lee(raw).strip()
        if not line:
            continue
        if line == "[ ... ]":
            lines.append((line, "omit"))
            continue
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            core = line[2:-2].strip()
            lines.append((core, "highlight"))
            continue
        lines.append((line, "normal"))
    return lines


def draw_text_panel(
    ax: plt.Axes,
    title: str,
    lines: List[Tuple[str, str]],
    face: str,
    edge: str,
) -> None:
    ax.set_axis_off()
    panel = patches.FancyBboxPatch(
        (0.015, 0.02),
        0.97,
        0.96,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.4,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(panel)

    ax.text(
        0.04,
        0.95,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        fontweight="bold",
        color="#0f172a",
    )

    draw_lines, body_font, line_h = layout_lines_for_axis(ax, lines)
    y = 0.89

    for text, kind in draw_lines:
        if y < 0.06:
            ax.text(
                0.05,
                0.05,
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
                fontsize=max(7.2, body_font - 0.8),
                style="italic",
                color="#64748b",
            )
        elif kind == "highlight":
            ax.text(
                0.05,
                y,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=body_font,
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
                fontsize=max(7.4, body_font - 0.2),
                color="#0f172a",
            )
        y -= line_h


def _jsonl_rows(path: Path) -> Iterable[Tuple[int, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield idx, obj


def load_run_text_map(input_jsonl: Path) -> Dict[str, Dict[str, str]]:
    """Map stable_key to full prompt/response text from run_all JSONL files."""
    result_dir = input_jsonl.parent.parent
    run_files = sorted(result_dir.glob("run_all_full_auto_attack_profile*.jsonl"))
    if not run_files:
        run_files = sorted(result_dir.glob("run_all_full_reframe_profile*.jsonl"))
    if not run_files:
        run_files = sorted(result_dir.glob("run_all_full_profile*.jsonl"))

    text_map: Dict[str, Dict[str, str]] = {}
    for run_file in run_files:
        for row_idx, payload in _jsonl_rows(run_file):
            stable_key = f"{run_file.name}::{row_idx}"
            prompt_text = str(payload.get("prompt_text") or payload.get("prompt") or "")
            response_text = str(payload.get("response") or payload.get("response_text") or "")
            text_map[stable_key] = {
                "prompt_text": prompt_text,
                "response_text": response_text,
            }
    return text_map


def clean_text(text: str) -> str:
    t = redact_lee(str(text or ""))
    t = t.replace("\r", "\n")
    t = t.replace("```", " ")
    t = t.replace("**", "")
    t = t.replace("*", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_sentences(text: str) -> List[str]:
    t = clean_text(text)
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    out = [p.strip() for p in parts if p.strip()]
    long_enough = [p for p in out if len(p) >= 18]
    return long_enough or out


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "with",
    "you",
    "your",
}


def tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"[a-zA-Z0-9']+", str(text or "").lower())
    return {tok for tok in tokens if len(tok) >= 4 and tok not in STOPWORDS}


def quoted_fragments(text: str) -> List[str]:
    frags: List[str] = []
    for quoted in re.findall(r"'([^']+)'", str(text or "")):
        q = re.sub(r"\s+", " ", quoted).strip(" .")
        if len(q) >= 8:
            frags.append(q)
    for quoted in re.findall(r"\"([^\"]+)\"", str(text or "")):
        q = re.sub(r"\s+", " ", quoted).strip(" .")
        if len(q) >= 8:
            frags.append(q)
    seen = set()
    out = []
    for frag in frags:
        key = frag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(frag)
    out.sort(key=len, reverse=True)
    return out


def bold_fragments(excerpt: str) -> List[str]:
    frags: List[str] = []
    for raw in str(excerpt or "").splitlines():
        line = raw.strip()
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            core = line[2:-2].strip()
            if len(core) >= 8:
                frags.append(core)
    return frags


def focus_fragments(reason: str, response_excerpt: str) -> List[str]:
    seen = set()
    out: List[str] = []
    for frag in quoted_fragments(reason) + bold_fragments(response_excerpt):
        key = frag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(frag)
    return out


def find_anchor_indices(sentences: List[str], fragments: List[str]) -> List[int]:
    if not sentences:
        return []
    if not fragments:
        return [0]

    frag_tokens = tokenize(" ".join(fragments))
    scored: List[Tuple[int, int]] = []

    for idx, sent in enumerate(sentences):
        sent_l = sent.lower()
        frag_match = any(
            frag.lower() in sent_l for frag in fragments if len(frag) >= 10
        )
        overlap = len(tokenize(sent) & frag_tokens)
        score = (3 if frag_match else 0) + overlap
        if frag_match or overlap >= 3:
            scored.append((score, idx))

    if not scored:
        fallback = sorted(
            ((len(tokenize(sent) & frag_tokens), idx) for idx, sent in enumerate(sentences)),
            key=lambda x: (-x[0], x[1]),
        )
        anchors = [idx for score, idx in fallback[:2] if score > 0]
        if anchors:
            return sorted(anchors)
        return [0]

    ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
    anchors: List[int] = []
    for _, idx in ranked:
        if idx not in anchors:
            anchors.append(idx)
        if len(anchors) >= 4:
            break
    return sorted(anchors)


def expand_indices(
    anchors: List[int],
    total_size: int,
    window: int,
    max_total: int,
) -> List[int]:
    if total_size <= 0:
        return []
    if not anchors:
        anchors = [0]

    candidate: Set[int] = set()
    for anchor in anchors:
        for idx in range(anchor - window, anchor + window + 1):
            if 0 <= idx < total_size:
                candidate.add(idx)
    if not candidate:
        candidate.add(0)

    if max_total > 0 and len(candidate) > max_total:
        scored = []
        for idx in candidate:
            dist = min(abs(idx - a) for a in anchors)
            anchor_rank = 0 if idx in anchors else 1
            scored.append((anchor_rank, dist, idx))
        scored.sort()
        candidate = {idx for _, _, idx in scored[:max_total]}

    return sorted(candidate)


def indices_to_lines(
    sentences: List[str],
    indices: List[int],
    highlight_indices: Set[int],
) -> List[Tuple[str, str]]:
    lines: List[Tuple[str, str]] = []
    prev = None
    for idx in indices:
        if prev is not None and idx - prev > 1:
            lines.append(("[ ... ]", "omit"))
        kind = "highlight" if idx in highlight_indices else "normal"
        lines.append((sentences[idx], kind))
        prev = idx
    return lines


def build_response_lines(
    full_response_text: str,
    response_excerpt: str,
    risky_reason: str,
    max_sentences: int,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    sentences = split_sentences(full_response_text)
    if not sentences:
        fallback = parse_excerpt_lines(response_excerpt)
        anchors = [line for line, kind in fallback if kind == "highlight"]
        return fallback, anchors

    fragments = focus_fragments(risky_reason, response_excerpt)
    anchors = find_anchor_indices(sentences, fragments)
    selected = expand_indices(
        anchors=anchors,
        total_size=len(sentences),
        window=1,
        max_total=max_sentences,
    )
    line_items = indices_to_lines(sentences, selected, set(anchors))
    anchor_text = [sentences[i] for i in anchors if 0 <= i < len(sentences)]
    return line_items, anchor_text


def build_prompt_lines(
    full_prompt_text: str,
    prompt_excerpt: str,
    risky_reason: str,
    response_anchor_text: List[str],
    max_sentences: int,
) -> List[Tuple[str, str]]:
    sentences = split_sentences(full_prompt_text)
    if not sentences:
        return parse_excerpt_lines(prompt_excerpt)

    focus_text = " ".join(response_anchor_text + quoted_fragments(risky_reason))
    focus_tokens = tokenize(focus_text)
    scored: List[Tuple[int, int]] = []
    for idx, sent in enumerate(sentences):
        overlap = len(tokenize(sent) & focus_tokens)
        scored.append((overlap, idx))

    seeds = [idx for overlap, idx in sorted(scored, key=lambda x: (-x[0], x[1])) if overlap >= 2][:3]
    if not seeds:
        seeds = [idx for overlap, idx in sorted(scored, key=lambda x: (-x[0], x[1])) if overlap > 0][:2]
    if not seeds:
        seeds = [0]

    selected = expand_indices(
        anchors=sorted(seeds),
        total_size=len(sentences),
        window=1,
        max_total=max_sentences,
    )
    return indices_to_lines(sentences, selected, set(seeds))


PANEL_LEFT = 0.05
PANEL_RIGHT = 0.95
PANEL_TOP = 0.89
PANEL_BOTTOM = 0.06


def _axis_size_pts(ax: plt.Axes) -> Tuple[float, float]:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=renderer)
    return bbox.width * 72.0 / fig.dpi, bbox.height * 72.0 / fig.dpi


def _rewrap_for_font(lines: List[Tuple[str, str]], max_chars: int) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    width = max(14, max_chars)
    for text, kind in lines:
        if kind == "omit":
            out.append((text, kind))
            continue
        wrapped = textwrap.wrap(str(text), width=width) or [str(text)]
        out.extend((w, kind) for w in wrapped)
    return out


def layout_lines_for_axis(ax: plt.Axes, lines: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], float, float]:
    ax_w_pts, ax_h_pts = _axis_size_pts(ax)
    avail_w_pts = max(1.0, (PANEL_RIGHT - PANEL_LEFT) * ax_w_pts)
    avail_h_pts = max(1.0, (PANEL_TOP - PANEL_BOTTOM) * ax_h_pts)

    for step in range(44, 27, -1):  # 11.0 down to 7.0
        body_font = step / 4.0
        max_chars = int(avail_w_pts / (0.56 * body_font))
        draw_lines = _rewrap_for_font(lines, max_chars=max_chars)
        line_step_pts = body_font * 1.23
        if len(draw_lines) * line_step_pts <= avail_h_pts:
            return draw_lines, body_font, line_step_pts / ax_h_pts

    body_font = 7.0
    max_chars = int(avail_w_pts / (0.56 * body_font))
    draw_lines = _rewrap_for_font(lines, max_chars=max_chars)
    line_step_pts = body_font * 1.23
    max_fit = max(1, int(avail_h_pts // max(line_step_pts, 1e-6)))
    if len(draw_lines) > max_fit:
        keep = max(1, max_fit - 1)
        draw_lines = draw_lines[:keep]
        if draw_lines[-1][0] != "[ ... ]":
            draw_lines.append(("[ ... ]", "omit"))
    return draw_lines, body_font, line_step_pts / ax_h_pts


def reason_lines(reason: str, width: int = 130) -> List[str]:
    cleaned = re.sub(r"\s+", " ", redact_lee(str(reason or ""))).strip()
    if not cleaned:
        return ["No risky reason provided."]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if not sentences:
        sentences = [cleaned]
    out: List[str] = []
    for sent in sentences[:6]:
        wrapped = textwrap.wrap(sent, width=width) or [sent]
        if wrapped:
            out.append(f"- {wrapped[0]}")
            out.extend(f"  {w}" for w in wrapped[1:])
    return out


def slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "").strip().lower())
    s = s.strip("_")
    return s or "unknown"


def iter_pairs(pairs: List[Dict], max_pairs: int) -> Iterable[Dict]:
    if max_pairs and max_pairs > 0:
        return pairs[:max_pairs]
    return pairs


def render_pair_figure(
    pair: Dict,
    run_text_map: Dict[str, Dict[str, str]],
    pattern_name: str,
    output_dir: Path,
    palette: Dict[str, str],
    palette_name: str,
    max_prompt_sentences: int,
    max_response_sentences: int,
    dpi: int,
) -> Tuple[Path, Path]:
    rank = int(pair.get("rank") or 0)
    prompt_key = str(pair.get("prompt_key") or "")
    stable_key = str(pair.get("stable_key") or "")
    pattern_score = pair.get("pattern_score")
    risk = pair.get("risk")
    safety = pair.get("safety")

    risky_reason = str(pair.get("risky_reason_focus") or "")

    full_prompt = run_text_map.get(stable_key, {}).get("prompt_text", "")
    full_response = run_text_map.get(stable_key, {}).get("response_text", "")

    response_lines, response_anchor_text = build_response_lines(
        full_response_text=full_response,
        response_excerpt=str(pair.get("response_excerpt", "")),
        risky_reason=risky_reason,
        max_sentences=max_response_sentences,
    )
    prompt_lines = build_prompt_lines(
        full_prompt_text=full_prompt,
        prompt_excerpt=str(pair.get("prompt_excerpt", "")),
        risky_reason=risky_reason,
        response_anchor_text=response_anchor_text,
        max_sentences=max_prompt_sentences,
    )
    why_lines = reason_lines(risky_reason, width=128)

    fig = plt.figure(figsize=(16.8, 10.6), facecolor=palette["fig_face"])
    gs = fig.add_gridspec(3, 2, height_ratios=[0.17, 0.64, 0.19], hspace=0.11, wspace=0.08)

    ax_h = fig.add_subplot(gs[0, :])
    ax_h.set_axis_off()
    ax_h.text(
        0.5,
        0.78,
        f"{pattern_name} Case Representation",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#0f172a",
    )
    meta = (
        f"Pair Rank {rank} | Prompt: {prompt_key} | Pattern Score: {pattern_score} | "
        f"Risk: {risk} | Safety: {safety}"
    )
    ax_h.text(0.5, 0.47, meta, ha="center", va="center", fontsize=12, color="#334155")
    ax_h.text(
        0.5,
        0.22,
        "Prompt and response highlights mark matched risky framing terms.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#64748b",
    )

    ax_l = fig.add_subplot(gs[1, 0])
    draw_text_panel(
        ax_l,
        "Prompt Excerpt (Matched Risk Cues Highlighted)",
        prompt_lines,
        face=palette["prompt_face"],
        edge=palette["prompt_edge"],
    )

    ax_r = fig.add_subplot(gs[1, 1])
    draw_text_panel(
        ax_r,
        "Response Excerpt (Risky Signals Highlighted)",
        response_lines,
        face=palette["resp_face"],
        edge=palette["resp_edge"],
    )

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
        0.84,
        f"Why This Pair Has a {pattern_name} Issue (Judge Risky Reason)",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color="#0f172a",
    )
    ax_b.text(
        0.03,
        0.65,
        "\n".join(why_lines),
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=10.8,
        color="#334155",
    )

    file_stem = (
        f"{slugify(pattern_name)}_pair_{rank:02d}_{slugify(prompt_key)}_{slugify(stable_key)}_{slugify(palette_name)}"
    )
    png_path = output_dir / f"{file_stem}.png"
    pdf_path = output_dir / f"{file_stem}.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


def main() -> int:
    args = parse_args()
    palette = get_palette(args.palette)
    pattern_obj = load_pattern_obj(args.input_jsonl, args.pattern)
    pairs = get_ranked_pairs(pattern_obj)
    if not pairs:
        raise ValueError(
            "No prompt/response pairs found. Expected one of: "
            "top10_prompt_response_pairs, top6_prompt_response_pairs, top3_prompt_response_pairs."
        )

    selected = list(iter_pairs(pairs, args.max_pairs))
    if not selected:
        raise ValueError("No pairs selected to render.")
    run_text_map = load_run_text_map(args.input_jsonl)

    for pair in selected:
        png_path, pdf_path = render_pair_figure(
            pair=pair,
            run_text_map=run_text_map,
            pattern_name=str(pattern_obj.get("codebook_pattern_name", args.pattern)),
            output_dir=args.output_dir,
            palette=palette,
            palette_name=args.palette,
            max_prompt_sentences=max(2, args.max_prompt_sentences),
            max_response_sentences=max(3, args.max_response_sentences),
            dpi=args.dpi,
        )
        print(png_path)
        print(pdf_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
