#!/usr/bin/env python3
"""Generate codebook markdown reports from judge artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


METRIC_ORDER: List[Tuple[str, str]] = [
    ("Risk", "Risk"),
    ("Safety", "Safety"),
    ("SinglePrompt", "SinglePrompt"),
    ("Sycophancy", "Sycophancy"),
    ("Validation", "Validation"),
    ("Elaboration", "Elaboration"),
    ("Behavioral", "Behavioral Advice"),
    ("Misrepresentation", "Misrepresentation"),
    ("Reality", "Reality Testing"),
    ("Concern", "Concern for Wellbeing"),
    ("Referral", "Referral"),
    ("DeEscalation", "De-escalation"),
    ("Warmth", "Relational Warmth"),
]


RISK_REASON_PATHS: List[Tuple[str, Tuple[str, ...]]] = [
    ("SinglePrompt", ("single_prompt_code",)),
    ("Validation", ("universal_codes", "validation")),
    ("Behavioral", ("universal_codes", "behavioral_advice")),
    ("Elaboration", ("universal_codes", "elaboration")),
    ("Misrepresentation", ("universal_codes", "misrepresentation")),
    ("Sycophancy", ("universal_codes", "sycophancy")),
]


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _to_float(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    return int(float(str(value)))


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).replace("```", "``\\`")


def _compact_whitespace(text: Any) -> str:
    return " ".join(str(text or "").split())


def _jsonl_rows(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                yield line_idx, payload


def infer_mode(result_dir: Path) -> str:
    return "auto_attack"


def report_title(mode: str) -> str:
    return "Auto Attack Results Codebook Report"


def load_run_map(result_dir: Path, mode: str) -> Dict[str, Dict[str, Any]]:
    run_map: Dict[str, Dict[str, Any]] = {}
    run_files = sorted(result_dir.glob(f"run_all_full_{mode}_profile*.jsonl"))
    for run_file in run_files:
        for line_idx, payload in _jsonl_rows(run_file):
            stable_key = f"{run_file.name}::{line_idx}"
            run_map[stable_key] = {
                "prompt_text": payload.get("prompt_text", ""),
                "response": payload.get("response", ""),
                "prompt": payload.get("prompt", ""),
            }
    return run_map


def load_judged_rows(result_dir: Path, mode: str) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    parse_error_count = 0
    judge_files = sorted(result_dir.glob(f"codebook_judge_results_full_{mode}_profile*.jsonl"))

    for judge_file in judge_files:
        for line_idx, payload in _jsonl_rows(judge_file):
            has_error = payload.get("error") is not None
            has_judgment = isinstance(payload.get("judgment"), dict)
            has_csv_row = isinstance(payload.get("csv_row"), dict)
            if has_error or not has_judgment or not has_csv_row:
                parse_error_count += 1
                continue

            csv_row = payload["csv_row"]
            rows.append(
                {
                    "stable_key": payload.get(
                        "stable_key",
                        f"{payload.get('source_file', judge_file.name)}::{payload.get('source_file_index', line_idx)}",
                    ),
                    "source_file": payload.get("source_file", judge_file.name),
                    "source_file_index": payload.get("source_file_index", line_idx),
                    "prompt": payload.get("prompt_raw") or payload.get("prompt", ""),
                    "risk": _to_float(csv_row.get("Risk")),
                    "safety": _to_float(csv_row.get("Safety")),
                    "single_prompt": _to_int(csv_row.get("SinglePrompt")),
                    "csv_row": csv_row,
                    "judgment": payload["judgment"],
                    "prompt_text": "",
                    "response_text": payload.get("response_text", ""),
                }
            )

    return rows, parse_error_count


def enrich_with_run_text(rows: List[Dict[str, Any]], run_map: Dict[str, Dict[str, Any]]) -> None:
    for row in rows:
        # Judge keys are "<file>::<line>" or "<file>::<line>::<response hash>".
        run_payload = run_map.get("::".join(str(row["stable_key"]).split("::")[:2]))
        if not run_payload:
            continue
        row["prompt_text"] = run_payload.get("prompt_text", "") or row["prompt_text"]
        row["response_text"] = run_payload.get("response", "") or row["response_text"]


def compute_prompt_stats(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["prompt"]].append(row)

    stats: List[Dict[str, Any]] = []
    for prompt, prompt_rows in grouped.items():
        stats.append(
            {
                "prompt": prompt,
                "n": len(prompt_rows),
                "mean_risk": _mean(r["risk"] for r in prompt_rows),
                "mean_safety": _mean(r["safety"] for r in prompt_rows),
            }
        )
    return stats


def get_risky_reason_lines(judgment: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for label, path in RISK_REASON_PATHS:
        node: Any = judgment
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if not isinstance(node, dict):
            continue
        score = _to_int(node.get("score"))
        if score <= 0:
            continue
        reason = _compact_whitespace(node.get("reason", ""))
        if not reason:
            continue
        lines.append(f"- {label}={score}: {reason}")
    return lines


def build_report(result_dir: Path, rows: List[Dict[str, Any]], parse_error_count: int, mode: str) -> str:
    overall_means: List[Tuple[str, float]] = []
    for csv_key, label in METRIC_ORDER:
        overall_means.append((label, _mean(_to_float(row["csv_row"].get(csv_key)) for row in rows)))

    prompt_stats = compute_prompt_stats(rows)
    highest_risk = sorted(
        prompt_stats,
        key=lambda item: (-item["mean_risk"], item["mean_safety"], item["prompt"]),
    )[:10]
    lowest_safety = sorted(
        prompt_stats,
        key=lambda item: (item["mean_safety"], -item["mean_risk"], item["prompt"]),
    )[:10]

    top_prompt_keys = [entry["prompt"] for entry in highest_risk[:5]]
    stats_lookup = {entry["prompt"]: entry for entry in prompt_stats}
    by_prompt: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt"]].append(row)

    lines: List[str] = []
    lines.append(f"# {report_title(mode)}")
    lines.append("")
    lines.append(f"Source directory: `{result_dir}`  ")
    lines.append(f"Judged rows analyzed: **{len(rows)}**  ")
    lines.append(f"Judge parse-error rows excluded: **{parse_error_count}**")
    lines.append("")

    lines.append("## Overall Codebook Pattern")
    lines.append("")
    lines.append("| Metric | Mean |")
    lines.append("|---|---:|")
    for label, mean_value in overall_means:
        lines.append(f"| {label} | {mean_value:.3f} |")
    lines.append("")

    lines.append("## Highest-risk Prompts (Mean Risk)")
    lines.append("")
    lines.append("| Rank | Prompt | N | Mean Risk | Mean Safety |")
    lines.append("|---:|---|---:|---:|---:|")
    for rank, item in enumerate(highest_risk, start=1):
        lines.append(
            f"| {rank} | {item['prompt']} | {item['n']} | {item['mean_risk']:.3f} | {item['mean_safety']:.3f} |"
        )
    lines.append("")

    lines.append("## Lowest-safety Prompts (Mean Safety)")
    lines.append("")
    lines.append("| Rank | Prompt | N | Mean Safety | Mean Risk |")
    lines.append("|---:|---|---:|---:|---:|")
    for rank, item in enumerate(lowest_safety, start=1):
        lines.append(
            f"| {rank} | {item['prompt']} | {item['n']} | {item['mean_safety']:.3f} | {item['mean_risk']:.3f} |"
        )
    lines.append("")

    lines.append("## Most Risky Five Prompt/Response Pairs For Each Highest-risk Prompt")
    lines.append("")

    for prompt_rank, prompt in enumerate(top_prompt_keys, start=1):
        stat = stats_lookup[prompt]
        lines.append(
            f"### {prompt_rank}. {prompt} (Mean Risk {stat['mean_risk']:.3f}, Mean Safety {stat['mean_safety']:.3f})"
        )
        lines.append("")

        ranked_pairs = sorted(
            by_prompt[prompt],
            key=lambda row: (-row["risk"], row["safety"], str(row["stable_key"])),
        )[:5]

        for pair_rank, pair in enumerate(ranked_pairs, start=1):
            lines.append(f"#### Pair {pair_rank}")
            lines.append(f"- Stable key: `{pair['stable_key']}`")
            lines.append(
                f"- Source: `{pair['source_file']}` row `{pair['source_file_index']}`"
            )
            lines.append(
                f"- Risk: `{pair['risk']}` | Safety: `{pair['safety']}` | SinglePrompt: `{pair['single_prompt']}`"
            )
            lines.append("")
            lines.append("**Risky reason (from judge)**")
            lines.append("")

            reason_lines = get_risky_reason_lines(pair["judgment"])
            if reason_lines:
                lines.extend(reason_lines)
            else:
                lines.append("- No risky dimensions with non-zero score were available in the parsed judgment.")
            lines.append("")

            prompt_text = _clean_text(pair.get("prompt_text")) or f"[Prompt text unavailable for `{pair['stable_key']}`]"
            response_text = _clean_text(pair.get("response_text")) or f"[Response text unavailable for `{pair['stable_key']}`]"

            lines.append("**Prompt**")
            lines.append("")
            lines.append("```text")
            lines.append(prompt_text)
            lines.append("```")
            lines.append("")
            lines.append("**Response**")
            lines.append("")
            lines.append("```text")
            lines.append(response_text)
            lines.append("```")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_report(result_dir: Path, mode: str | None = None, output_path: Path | None = None) -> Path:
    actual_mode = mode or infer_mode(result_dir)
    run_map = load_run_map(result_dir, mode=actual_mode)
    rows, parse_error_count = load_judged_rows(result_dir, mode=actual_mode)
    enrich_with_run_text(rows, run_map)

    if output_path is None:
        output_path = result_dir / "summary" / f"{actual_mode}_codebook_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(result_dir, rows, parse_error_count, mode=actual_mode), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dirs",
        nargs="+",
        help="One or more result directories containing judge/run JSONL files.",
    )
    parser.add_argument(
        "--mode",
        choices=["auto_attack"],
        help="Context mode (only auto_attack).",
    )
    parser.add_argument(
        "--output",
        help="Optional explicit output path (only valid when exactly one result directory is provided).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output and len(args.result_dirs) != 1:
        raise SystemExit("--output can only be used with a single result directory")

    for result_dir_raw in args.result_dirs:
        result_dir = Path(result_dir_raw)
        output_path = Path(args.output) if args.output else None
        written = write_report(result_dir=result_dir, mode=args.mode, output_path=output_path)
        print(str(written))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
