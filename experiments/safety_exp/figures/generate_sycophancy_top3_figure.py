#!/usr/bin/env python3
"""Generate paper-style figures for top prompt/response excerpts of one pattern."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib import patches


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
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument(
        "--stable-keys",
        nargs="*",
        default=None,
        help="Optional ordered stable_key list to prioritize when selecting pairs.",
    )
    parser.add_argument(
        "--pairs-per-fig",
        type=int,
        default=2,
        help="How many prompt/response pairs to place in one figure.",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("outputs/safety_exp/fig/fig4/sycophancy_top3_prompt_response.png"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("outputs/safety_exp/fig/fig4/sycophancy_top3_prompt_response.pdf"),
    )
    parser.add_argument(
        "--palette",
        default="peach_mint",
        choices=[
            "peach_mint",
            "rose_teal",
            "butter_sky",
            "cream_indigo",
        ],
        help="Color palette for prompt/response panels.",
    )
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
    """Load ranked pairs from top6/top3 schema variants."""
    for key in ("top10_prompt_response_pairs", "top6_prompt_response_pairs", "top3_prompt_response_pairs"):
        pairs = pattern_obj.get(key, [])
        if isinstance(pairs, list) and pairs:
            return list(pairs)
    return []


def parse_excerpt_lines(excerpt: str, wrap_width: int, response_only_bold: bool = False) -> List[Tuple[str, str]]:
    """Return (line, kind) where kind in {'normal','bold','omit'}."""
    out: List[Tuple[str, str]] = []
    for raw in str(excerpt or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == "[ ... ]":
            out.append((line, "omit"))
            continue

        is_bold = line.startswith("**") and line.endswith("**") and len(line) > 4
        if is_bold:
            core = line[2:-2].strip()
            wrapped = textwrap.wrap(core, width=wrap_width) or [core]
            out.extend((w, "bold") for w in wrapped)
        else:
            if response_only_bold:
                # For response panel, suppress non-bold content per user request.
                continue
            wrapped = textwrap.wrap(line, width=wrap_width) or [line]
            out.extend((w, "normal") for w in wrapped)
    return out


def _jsonl_rows(path: Path) -> Iterable[Tuple[int, Dict]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                payload = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield idx, payload


def load_run_response_map(input_jsonl: Path) -> Dict[str, str]:
    """Map stable_key -> full response text from run_all JSONL files."""
    # input_jsonl: .../results/<mode>/gpt4o/summary/*.jsonl
    result_dir = input_jsonl.parent.parent
    run_files = sorted(result_dir.glob("run_all_full_auto_attack_profile*.jsonl"))
    if not run_files:
        run_files = sorted(result_dir.glob("run_all_full_reframe_profile*.jsonl"))
    if not run_files:
        run_files = sorted(result_dir.glob("run_all_full_profile*.jsonl"))

    out: Dict[str, str] = {}
    for run_file in run_files:
        for idx, payload in _jsonl_rows(run_file):
            stable_key = f"{run_file.name}::{idx}"
            out[stable_key] = str(payload.get("response") or "")
    return out


def clean_text(text: str) -> str:
    t = str(text or "")
    t = t.replace("\r", " ")
    t = t.replace("```", " ")
    t = t.replace("**", "")
    t = t.replace("*", "")
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"^\s*[-]\s+", "", t, flags=re.M)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_sentences(text: str) -> List[str]:
    txt = clean_text(text)
    if not txt:
        return []
    sents = re.split(r"(?<=[.!?])\s+", txt)
    out: List[str] = []
    for s in sents:
        s = s.strip(" \t\n-")
        if len(s) < 28:
            continue
        out.append(s)
    return out


def trim_sentence(s: str, max_len: int = 220) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def reason_fragments(reason: str) -> List[str]:
    frags = []
    for q in re.findall(r"'([^']+)'", reason or ""):
        q = re.sub(r"\s+", " ", q).strip(" .")
        if len(q) >= 10:
            frags.append(q)
    for q in re.findall(r"\"([^\"]+)\"", reason or ""):
        q = re.sub(r"\s+", " ", q).strip(" .")
        if len(q) >= 10:
            frags.append(q)
    seen = set()
    dedup = []
    for q in frags:
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        dedup.append(q)
    dedup.sort(key=len, reverse=True)
    return dedup


def build_response_from_raw(full_response: str, reason_text: str) -> str:
    """Create a readable response excerpt with complete risky-signal sentences."""
    sents = split_sentences(full_response)
    if not sents:
        return "[Response text unavailable.]"

    selected: List[str] = []
    seen = set()

    # Prefer sentences that match quoted fragments from risky reason.
    for frag in reason_fragments(reason_text):
        fl = frag.lower()
        for s in sents:
            sl = s.lower()
            if fl in sl and sl not in seen:
                selected.append(trim_sentence(s))
                seen.add(sl)
                break
        if len(selected) >= 4:
            break

    # Fallback via risk keywords.
    if len(selected) < 3:
        keywords = [
            "right", "extraordinary", "clarity", "transcend", "simulation", "root access",
            "admin", "watcher", "dissolve", "graduation", "mission", "recruit", "signal",
        ]
        for s in sents:
            sl = s.lower()
            if sl in seen:
                continue
            if any(k in sl for k in keywords):
                selected.append(trim_sentence(s))
                seen.add(sl)
            if len(selected) >= 4:
                break

    # Final fallback.
    if len(selected) < 3:
        for s in sents:
            sl = s.lower()
            if sl in seen:
                continue
            selected.append(trim_sentence(s))
            seen.add(sl)
            if len(selected) >= 3:
                break

    lines: List[str] = []
    display = selected[:4]
    for i, s in enumerate(display):
        lines.append(f"**{s}**")
        if i < len(display) - 1:
            lines.append("[ ... ]")
    return "\n".join(lines)


def truncate_lines(lines: List[Tuple[str, str]], max_lines: int) -> List[Tuple[str, str]]:
    if len(lines) <= max_lines:
        return lines
    keep = max(1, max_lines - 1)
    out = list(lines[:keep])
    if out[-1][0] != "[ ... ]":
        out.append(("[ ... ]", "omit"))
    return out


def text_signature(pair: Dict) -> Set[str]:
    blob = " ".join(
        [
            str(pair.get("prompt_excerpt", "")),
            str(pair.get("response_excerpt", "")),
            str(pair.get("risky_reason_focus", "")),
        ]
    ).lower()
    tokens = re.findall(r"[a-z0-9]+", blob)
    return {t for t in tokens if len(t) >= 4}


def pair_identity(pair: Dict) -> str:
    prompt = re.sub(r"\s+", " ", str(pair.get("prompt_excerpt", "")).strip().lower())
    resp = re.sub(r"\s+", " ", str(pair.get("response_excerpt", "")).strip().lower())
    return f"{prompt}|||{resp}"


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def select_distinct_pairs(pairs: List[Dict], top_n: int) -> List[Dict]:
    """Select up to top_n pairs with stable_key uniqueness and textual diversity."""
    if top_n <= 0:
        return []
    if not pairs:
        return []

    selected: List[Dict] = []
    seen_keys: Set[str] = set()
    seen_identity: Set[str] = set()

    for pair in pairs:
        key = str(pair.get("stable_key", ""))
        if key in seen_keys:
            continue
        selected.append(pair)
        seen_keys.add(key)
        seen_identity.add(pair_identity(pair))
        break

    while len(selected) < top_n:
        best_pair: Optional[Dict] = None
        best_score = -1.0
        selected_sigs = [text_signature(p) for p in selected]
        selected_prompts = {
            str(p.get("prompt_key", "")).strip().lower()
            for p in selected
            if str(p.get("prompt_key", "")).strip()
        }
        # Prefer a new prompt key inside the same figure whenever available.
        need_new_prompt = False
        if selected_prompts:
            for p in pairs:
                cand_key = str(p.get("stable_key", ""))
                if cand_key in seen_keys:
                    continue
                cand_identity = pair_identity(p)
                if cand_identity in seen_identity:
                    continue
                cand_prompt = str(p.get("prompt_key", "")).strip().lower()
                if cand_prompt and cand_prompt not in selected_prompts:
                    need_new_prompt = True
                    break
        for candidate in pairs:
            key = str(candidate.get("stable_key", ""))
            if key in seen_keys:
                continue
            identity = pair_identity(candidate)
            if identity in seen_identity:
                continue
            cand_prompt = str(candidate.get("prompt_key", "")).strip().lower()
            if need_new_prompt and (not cand_prompt or cand_prompt in selected_prompts):
                continue
            cand_sig = text_signature(candidate)
            # Prefer candidates least similar to already selected ones.
            max_sim = max((jaccard(cand_sig, s) for s in selected_sigs), default=0.0)
            diversity = 1.0 - max_sim
            prompt_bonus = 0.35 if cand_prompt and cand_prompt not in selected_prompts else 0.0
            score = diversity + prompt_bonus
            if score > best_score:
                best_score = score
                best_pair = candidate
        if best_pair is None:
            break
        selected.append(best_pair)
        seen_keys.add(str(best_pair.get("stable_key", "")))
        seen_identity.add(pair_identity(best_pair))

    return selected


PANEL_LEFT_PAD = 0.055
PANEL_RIGHT_PAD = 0.955
PANEL_BODY_TOP = 0.865
PANEL_BODY_BOTTOM = 0.06


def _rewrap_for_font(src: List[Tuple[str, str]], max_chars: int) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    width = max(12, max_chars)
    for text, kind in src:
        if kind == "omit":
            out.append((text, kind))
            continue
        wrapped = textwrap.wrap(str(text), width=width) or [str(text)]
        out.extend((w, kind) for w in wrapped)
    return out


def _axis_size_pts(ax: plt.Axes) -> Tuple[float, float]:
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer=renderer)
    return bbox.width * 72.0 / fig.dpi, bbox.height * 72.0 / fig.dpi


def _fit_body_font(ax: plt.Axes, lines: List[Tuple[str, str]]) -> float:
    ax_w_pts, ax_h_pts = _axis_size_pts(ax)
    avail_w_pts = max(1.0, (PANEL_RIGHT_PAD - PANEL_LEFT_PAD) * ax_w_pts)
    avail_h_pts = max(1.0, (PANEL_BODY_TOP - PANEL_BODY_BOTTOM) * ax_h_pts)
    for step in range(64, 31, -1):  # 16.0 down to 8.0 by 0.25
        fs = step / 4.0
        max_chars = int(avail_w_pts / (0.57 * fs))
        wrapped = _rewrap_for_font(lines, max_chars)
        step_pts = fs * 1.22
        if len(wrapped) * step_pts <= avail_h_pts:
            return fs
    return 8.0


def _layout_for_font(
    ax: plt.Axes,
    lines: List[Tuple[str, str]],
    body_font: float,
) -> Tuple[List[Tuple[str, str]], float]:
    ax_w_pts, ax_h_pts = _axis_size_pts(ax)
    avail_w_pts = max(1.0, (PANEL_RIGHT_PAD - PANEL_LEFT_PAD) * ax_w_pts)
    avail_h_pts = max(1.0, (PANEL_BODY_TOP - PANEL_BODY_BOTTOM) * ax_h_pts)
    max_chars = int(avail_w_pts / (0.57 * body_font))
    draw_lines = _rewrap_for_font(lines, max_chars)
    step_pts = body_font * 1.22
    max_fit = max(1, int(avail_h_pts // max(step_pts, 1e-6)))
    if len(draw_lines) > max_fit:
        keep = max(1, max_fit - 1)
        draw_lines = draw_lines[:keep]
        if draw_lines[-1][0] != "[ ... ]":
            draw_lines.append(("[ ... ]", "omit"))
    line_h = step_pts / ax_h_pts
    return draw_lines, line_h


def draw_panel(
    ax: plt.Axes,
    title: str,
    lines: List[Tuple[str, str]],
    face: str,
    edge: str,
    body_font: float,
) -> None:
    ax.set_axis_off()
    panel = patches.FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.6,
        edgecolor=edge,
        facecolor=face,
        transform=ax.transAxes,
    )
    ax.add_patch(panel)

    draw_lines, line_h = _layout_for_font(ax, lines, body_font=body_font)
    ax.text(
        PANEL_LEFT_PAD,
        0.945,
        title,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=min(16.0, body_font + 1.5),
        fontweight="bold",
        color="#0f172a",
        clip_on=True,
        clip_path=panel,
    )

    y = PANEL_BODY_TOP
    for text, kind in draw_lines:
        if y < PANEL_BODY_BOTTOM:
            break
        if kind == "omit":
            ax.text(
                PANEL_LEFT_PAD,
                y,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=max(7.5, body_font - 1.0),
                style="italic",
                color="#64748b",
                clip_on=True,
                clip_path=panel,
            )
        elif kind == "bold":
            ax.text(
                PANEL_LEFT_PAD,
                y,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=body_font,
                fontweight="bold",
                color="#0f172a",
                clip_on=True,
                clip_path=panel,
            )
        else:
            ax.text(
                PANEL_LEFT_PAD,
                y,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=max(7.5, body_font - 0.5),
                color="#0f172a",
                clip_on=True,
                clip_path=panel,
            )
        y -= line_h


def chunk_list(items: List[Dict], n: int) -> List[List[Dict]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def output_variant(path: Path, idx: int, total: int) -> Path:
    if total <= 1:
        return path
    return path.with_name(f"{path.stem}_part{idx}{path.suffix}")


def choose_best_width_ratio(
    num_rows: int,
    row_lines: List[Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]],
    fig_h: float,
) -> float:
    """Pick prompt/response width split that maximizes shared readable font size."""
    candidates = [r / 100.0 for r in range(34, 67, 2)]
    preferred = 0.46
    best_ratio = preferred
    best_font = -1.0

    for prompt_ratio in candidates:
        fig_tmp = plt.figure(figsize=(17.0, fig_h))
        gs_tmp = fig_tmp.add_gridspec(
            num_rows,
            2,
            width_ratios=[prompt_ratio, 1.0 - prompt_ratio],
            hspace=0.14,
            wspace=0.075,
        )

        specs: List[Tuple[plt.Axes, List[Tuple[str, str]]]] = []
        for i, (prompt_lines, response_lines) in enumerate(row_lines, start=1):
            ax_prompt = fig_tmp.add_subplot(gs_tmp[i - 1, 0])
            ax_resp = fig_tmp.add_subplot(gs_tmp[i - 1, 1])
            specs.append((ax_prompt, prompt_lines))
            specs.append((ax_resp, response_lines))

        candidate_font = min(_fit_body_font(ax, lines) for ax, lines in specs)
        plt.close(fig_tmp)

        if candidate_font > best_font + 1e-6:
            best_font = candidate_font
            best_ratio = prompt_ratio
        elif abs(candidate_font - best_font) <= 1e-6:
            if abs(prompt_ratio - preferred) < abs(best_ratio - preferred):
                best_ratio = prompt_ratio

    return best_ratio


def main() -> int:
    args = parse_args()
    palette = get_palette(args.palette)
    obj = load_pattern_obj(args.input_jsonl, args.pattern)
    all_pairs = get_ranked_pairs(obj)
    if args.stable_keys:
        pair_by_key = {str(p.get("stable_key", "")): p for p in all_pairs}
        selected: List[Dict] = []
        seen = set()
        for key in args.stable_keys:
            k = str(key)
            if k in pair_by_key and k not in seen:
                selected.append(pair_by_key[k])
                seen.add(k)
        for p in all_pairs:
            k = str(p.get("stable_key", ""))
            if k in seen:
                continue
            selected.append(p)
        all_pairs = selected

    pairs = select_distinct_pairs(all_pairs, max(args.top_n, 1))
    if not pairs:
        raise ValueError(
            "No prompt/response pairs found for selected pattern. "
            "Expected one of: top6_prompt_response_pairs, top3_prompt_response_pairs."
        )
    if len(pairs) < max(args.top_n, 1):
        print(
            f"Warning: requested top_n={args.top_n} but only {len(pairs)} distinct pairs available."
        )

    response_map = load_run_response_map(args.input_jsonl)

    pages = chunk_list(pairs, max(1, args.pairs_per_fig))
    for page_idx, page_pairs in enumerate(pages, start=1):
        fig_h = 1.1 + 4.9 * len(page_pairs)
        row_data: List[Dict[str, Any]] = []
        for i, pair in enumerate(page_pairs, start=1):
            prompt_title = f"Prompt {i}"
            response_title = f"Response {i}"

            prompt_lines = parse_excerpt_lines(pair.get("prompt_excerpt", ""), wrap_width=220, response_only_bold=False)
            prompt_lines = truncate_lines(prompt_lines, max_lines=9)

            full_response = response_map.get(pair.get("stable_key", ""), "")
            response_excerpt = build_response_from_raw(full_response, str(pair.get("risky_reason_focus", "")))
            response_lines = parse_excerpt_lines(response_excerpt, wrap_width=260, response_only_bold=True)
            response_lines = truncate_lines(response_lines, max_lines=9)
            if not response_lines:
                response_lines = [("[No highlighted risky sentence found.]", "normal")]

            row_data.append(
                {
                    "prompt_title": prompt_title,
                    "response_title": response_title,
                    "prompt_lines": prompt_lines,
                    "response_lines": response_lines,
                }
            )

        width_ratio = choose_best_width_ratio(
            num_rows=len(page_pairs),
            row_lines=[(r["prompt_lines"], r["response_lines"]) for r in row_data],
            fig_h=fig_h,
        )

        fig = plt.figure(figsize=(17.0, fig_h), facecolor=palette["fig_face"])
        gs = fig.add_gridspec(
            len(page_pairs),
            2,
            width_ratios=[width_ratio, 1.0 - width_ratio],
            hspace=0.14,
            wspace=0.075,
        )

        panel_specs: List[Tuple[plt.Axes, str, List[Tuple[str, str]], str, str]] = []
        for i, row in enumerate(row_data, start=1):
            ax_prompt = fig.add_subplot(gs[i - 1, 0])
            ax_resp = fig.add_subplot(gs[i - 1, 1])

            panel_specs.append(
                (
                    ax_prompt,
                    row["prompt_title"],
                    row["prompt_lines"],
                    palette["prompt_face"],
                    palette["prompt_edge"],
                )
            )
            panel_specs.append(
                (
                    ax_resp,
                    row["response_title"],
                    row["response_lines"],
                    palette["resp_face"],
                    palette["resp_edge"],
                )
            )

        shared_body_font = min(_fit_body_font(ax, lines) for ax, _, lines, _, _ in panel_specs)
        shared_body_font = max(8.0, shared_body_font)

        for ax, title, lines, face, edge in panel_specs:
            draw_panel(
                ax,
                title,
                lines,
                face=face,
                edge=edge,
                body_font=shared_body_font,
            )

        png_path = output_variant(args.output_png, page_idx, len(pages))
        pdf_path = output_variant(args.output_pdf, page_idx, len(pages))
        for out in [png_path, pdf_path]:
            out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        print(png_path)
        print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
