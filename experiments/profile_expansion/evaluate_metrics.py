#!/usr/bin/env python3
"""Evaluate profile-expansion outputs with all implemented metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from angel_common.paths import REPO_ROOT

PACKAGE_ROOT = REPO_ROOT  # repository root; run scripts with `python -m` from here

from experiments.profile_expansion.evaluate_metrics_core import (
    build_behavior_section,
    build_profile_alignment_section,
    build_semantic_section,
    compute_records_signature,
    evaluate,
    load_resume_state,
    read_jsonl,
    write_json,
)
from experiments.profile_expansion.metrics.semantic_diversity_knn import (
    score_group_diversity,
)
from experiments.profile_expansion.evaluate_metrics_record_cache import (
    load_record_cache,
    record_cache_key,
    save_record_cache,
)
from experiments.profile_expansion.evaluate_metrics_runs import (
    derive_run_number_distributions,
    filter_records_by_runs,
    format_run_counts,
    resolve_requested_runs,
)

SUPPORTED_METRICS = (
    "profile_alignment",
    "semantic_diversity",
    "behavior_diversity",
    "group_diversity",
    "min_distance_diversity",
)


def parse_metrics_arg(raw: str) -> Set[str]:
    metrics: Set[str] = set()
    for token in (raw or "").split(","):
        item = token.strip()
        if not item:
            continue
        metrics.add(item)
    if not metrics:
        raise ValueError("--metrics must specify at least one metric.")
    invalid = sorted(metrics - set(SUPPORTED_METRICS))
    if invalid:
        raise ValueError(
            f"Unsupported metric(s): {invalid}. "
            f"Supported metrics: {list(SUPPORTED_METRICS)}"
        )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate profile-expansion JSONL outputs.")
    parser.add_argument("--input", required=True, help="Experiment output JSONL.")
    parser.add_argument("--output", required=True, help="Metrics JSON output path.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path for incremental saves (default: <output>.checkpoint.json).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write checkpoint every N profile-alignment records (default: 1).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint state if available and compatible.",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        action="append",
        default=None,
        help=(
            "Run number to evaluate (1-based within each profile/model). "
            "Repeat for multiple values, e.g. --run-number 3 --run-number 6."
        ),
    )
    parser.add_argument(
        "--run-numbers",
        default=None,
        help="Comma-separated run numbers, e.g. --run-numbers 3,4,5,6.",
    )
    parser.add_argument(
        "--run-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        default=None,
        help=(
            "Prefix run-count sweep. "
            "--run-range K K evaluates first K runs per profile. "
            "--run-range A B evaluates first K runs per profile for every K in [A..B] "
            "and stores each result."
        ),
    )
    parser.add_argument(
        "--record-cache",
        default=None,
        help=(
            "Path to persistent per-record cache for expensive judge/extraction steps. "
            "Default: <input>.record_cache.json (unless --no-record-cache)."
        ),
    )
    parser.add_argument(
        "--no-record-cache",
        action="store_true",
        help="Disable persistent per-record cache.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Max concurrent workers for behavior-diversity LLM extraction. >1 runs an "
            "adaptive pre-extraction pass (AIMD): starts at this many in-flight calls, "
            "backs off on rate-limit (429) errors, and ramps back up while healthy. Default: 1."
        ),
    )
    parser.add_argument(
        "--workers-decrease-factor",
        type=float,
        default=0.5,
        help=(
            "AIMD multiplicative decrease factor applied to concurrency on a rate-limit hit "
            "(0<f<1; smaller = more aggressive backoff). Default: 0.5."
        ),
    )
    parser.add_argument(
        "--workers-cooldown",
        type=float,
        default=3.0,
        help=(
            "AIMD cooldown in seconds; rate-limit signals within this window of the last "
            "decrease are treated as one event (debounces bursts). Default: 3.0."
        ),
    )
    parser.add_argument(
        "--workers-success-step",
        type=int,
        default=0,
        help=(
            "AIMD additive-increase cadence: number of successful extractions per +1 to "
            "concurrency. 0 = auto (= --workers, i.e. ramp up once per generation). Default: 0."
        ),
    )
    parser.add_argument(
        "--metrics",
        default="semantic_diversity,behavior_diversity,group_diversity",
        help=(
            "Comma-separated metrics to compute. "
            "Supported: profile_alignment,semantic_diversity,behavior_diversity,group_diversity. "
            "Default: semantic_diversity,behavior_diversity,group_diversity."
        ),
    )
    return parser.parse_args()


def summarize_report_for_log(report: Dict[str, Any]) -> str:
    parts: List[str] = []
    profile = report.get("profile_alignment")
    if isinstance(profile, dict):
        if profile.get("score") is not None:
            parts.append(f"profile_alignment_norm={float(profile['score']):.3f}")
        if profile.get("avg_score_1_to_5") is not None:
            parts.append(f"profile_alignment_avg_1_to_5={float(profile['avg_score_1_to_5']):.3f}")
    semantic = report.get("semantic_diversity")
    if isinstance(semantic, dict):
        if semantic.get("score") is not None:
            parts.append(f"semantic_diversity={float(semantic['score']):.3f}")
        if semantic.get("score_raw") is not None:
            parts.append(f"semantic_diversity_raw={float(semantic['score_raw']):.3f}")
    behavior = report.get("behavior_diversity")
    if isinstance(behavior, dict) and behavior.get("score") is not None:
        parts.append(f"behavior_diversity={float(behavior['score']):.3f}")
    group = report.get("group_diversity")
    if isinstance(group, dict) and group.get("score") is not None:
        parts.append(f"group_diversity={float(group['score']):.3f}")
    min_distance = report.get("min_distance_diversity")
    if isinstance(min_distance, dict) and min_distance.get("score") is not None:
        parts.append(f"min_distance_diversity={float(min_distance['score']):.3f}")
    return ", ".join(parts) if parts else "(no metric sections)"


def derive_per_k_output_path(base_output: Path, k: int) -> Path:
    return base_output.with_name(f"{base_output.stem}.run{k}{base_output.suffix}")


def derive_per_k_checkpoint_path(base_checkpoint: Path, k: int) -> Path:
    return base_checkpoint.with_name(f"{base_checkpoint.stem}.run{k}{base_checkpoint.suffix}")


def evaluate_selected_records(
    *,
    records: List[Dict[str, Any]],
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    checkpoint_every: int,
    resume: bool,
    profile_alignment_cache: Optional[Dict[str, Dict[str, Any]]],
    behavior_extraction_cache: Optional[Dict[str, Dict[str, Any]]],
    semantic_embedding_cache: Optional[Dict[str, Any]],
    cache_stats: Dict[str, int],
    include_profile_alignment: bool,
    include_semantic_diversity: bool,
    include_behavior_diversity: bool,
    include_group_diversity: bool,
    include_min_distance_diversity: bool,
    behavior_max_workers: int = 1,
    behavior_worker_tuning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    input_signature = compute_records_signature(records)

    resume_alignment: List[Dict[str, Any]] = []
    resume_semantic: List[Dict[str, Any]] = []
    resume_behavior: List[Dict[str, Any]] = []
    resume_min_distance: List[Dict[str, Any]] = []

    if resume:
        resume_alignment, resume_semantic, resume_behavior, resume_min_distance = load_resume_state(
            checkpoint_path=checkpoint_path,
            input_signature=input_signature,
            num_records=len(records),
        )
        if resume_alignment or resume_semantic or resume_behavior or resume_min_distance:
            print(
                "[Checkpoint] Resume loaded: "
                f"alignment={len(resume_alignment)}, "
                f"semantic_profiles={len(resume_semantic)}, "
                f"behavior_profiles={len(resume_behavior)}, "
                f"min_distance_profiles={len(resume_min_distance)}",
                flush=True,
            )
    if not include_profile_alignment:
        resume_alignment = []
    if not include_semantic_diversity:
        resume_semantic = []
    if not include_behavior_diversity:
        resume_behavior = []
    if not include_min_distance_diversity:
        resume_min_distance = []

    report = evaluate(
        records,
        input_path=input_path,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
        input_signature=input_signature,
        checkpoint_every=checkpoint_every,
        resume_alignment_by_record=resume_alignment,
        resume_semantic_profiles=resume_semantic,
        resume_behavior_profiles=resume_behavior,
        resume_min_distance_profiles=resume_min_distance,
        profile_alignment_cache=profile_alignment_cache,
        behavior_extraction_cache=behavior_extraction_cache,
        semantic_embedding_cache=semantic_embedding_cache,
        cache_stats=cache_stats,
        record_cache_key_fn=record_cache_key,
        compute_profile_alignment=include_profile_alignment,
        compute_semantic_diversity=include_semantic_diversity,
        compute_behavior_diversity=include_behavior_diversity,
        compute_min_distance_diversity=include_min_distance_diversity,
        behavior_max_workers=behavior_max_workers,
        behavior_worker_tuning=behavior_worker_tuning,
    )

    if include_group_diversity:
        print("[GroupDiversity] enabled; computing additional metric", flush=True)
        group_diversity = score_group_diversity(
            records,
            embedding_cache=semantic_embedding_cache,
            embedding_cache_stats=cache_stats,
        )
        report["group_diversity"] = group_diversity

    write_json(output_path, report)
    return report


def filter_for_requested_runs(
    *,
    all_records: List[Dict[str, Any]],
    requested_runs: Set[int],
) -> List[Dict[str, Any]]:
    records, selected_distribution, all_distribution = filter_records_by_runs(
        all_records,
        requested_runs=requested_runs,
    )
    print(
        "[RunFilter] Requested runs="
        f"{sorted(requested_runs)}; "
        f"selected {len(records)}/{len(all_records)} records. "
        f"Selected distribution: {format_run_counts(selected_distribution)}. "
        f"All runs before filter: {format_run_counts(all_distribution)}",
        flush=True,
    )
    if not records:
        raise ValueError(
            "Run filter removed all records. "
            f"Requested runs={sorted(requested_runs)}; available={format_run_counts(all_distribution)}"
        )
    return records


def run_prefix_sweep(
    *,
    ks: Sequence[int],
    all_records: List[Dict[str, Any]],
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    checkpoint_every: int,
    resume: bool,
    profile_alignment_cache: Optional[Dict[str, Dict[str, Any]]],
    behavior_extraction_cache: Optional[Dict[str, Dict[str, Any]]],
    semantic_embedding_cache: Optional[Dict[str, Any]],
    cache_stats: Dict[str, int],
    include_profile_alignment: bool,
    include_semantic_diversity: bool,
    include_behavior_diversity: bool,
    include_group_diversity: bool,
    include_min_distance_diversity: bool,
    behavior_max_workers: int = 1,
    behavior_worker_tuning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    all_distribution, _ = derive_run_number_distributions(all_records)
    results: List[Dict[str, Any]] = []

    for k in ks:
        requested_runs = set(range(1, k + 1))
        print(f"[RunSweep] Evaluating prefix runs 1..{k}", flush=True)
        records = filter_for_requested_runs(
            all_records=all_records,
            requested_runs=requested_runs,
        )

        per_k_output = derive_per_k_output_path(output_path, k)
        per_k_checkpoint = derive_per_k_checkpoint_path(checkpoint_path, k)

        report = evaluate_selected_records(
            records=records,
            input_path=input_path,
            output_path=per_k_output,
            checkpoint_path=per_k_checkpoint,
            checkpoint_every=checkpoint_every,
            resume=resume,
            profile_alignment_cache=profile_alignment_cache,
            behavior_extraction_cache=behavior_extraction_cache,
            semantic_embedding_cache=semantic_embedding_cache,
            cache_stats=cache_stats,
            include_profile_alignment=include_profile_alignment,
            include_semantic_diversity=include_semantic_diversity,
            include_behavior_diversity=include_behavior_diversity,
            include_group_diversity=include_group_diversity,
            include_min_distance_diversity=include_min_distance_diversity,
            behavior_max_workers=behavior_max_workers,
            behavior_worker_tuning=behavior_worker_tuning,
        )

        print(
            f"[RunSweep] Wrote k={k}: {per_k_output} "
            f"({summarize_report_for_log(report)})",
            flush=True,
        )

        results.append(
            {
                "k": k,
                "requested_runs": sorted(requested_runs),
                "num_records": report.get("num_records"),
                "output_path": str(per_k_output),
                "checkpoint_path": str(per_k_checkpoint),
                "profile_alignment_score": report.get("profile_alignment", {}).get("score"),
                "profile_alignment_avg_score_1_to_5": report.get("profile_alignment", {}).get("avg_score_1_to_5"),
                "semantic_diversity_score": report.get("semantic_diversity", {}).get("score"),
                "semantic_diversity_score_raw": report.get("semantic_diversity", {}).get("score_raw"),
                "behavior_diversity_score": report.get("behavior_diversity", {}).get("score"),
                "group_diversity_score": report.get("group_diversity", {}).get("score"),
                "min_distance_diversity_score": report.get("min_distance_diversity", {}).get("score"),
            }
        )

    manifest = {
        "mode": "run_range_prefix_sweep",
        "input_path": str(input_path),
        "base_output_path": str(output_path),
        "all_runs_distribution": all_distribution,
        "results": results,
    }
    write_json(output_path, manifest)
    print(f"[RunSweep] Wrote manifest: {output_path}", flush=True)
    return manifest


def main() -> None:
    args = parse_args()
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be >= 1")

    selected_metrics = parse_metrics_arg(args.metrics)
    include_profile_alignment = "profile_alignment" in selected_metrics
    include_semantic_diversity = "semantic_diversity" in selected_metrics
    include_behavior_diversity = "behavior_diversity" in selected_metrics
    include_group_diversity = "group_diversity" in selected_metrics
    include_min_distance_diversity = "min_distance_diversity" in selected_metrics
    behavior_worker_tuning = {
        "workers_decrease_factor": args.workers_decrease_factor,
        "workers_cooldown_seconds": args.workers_cooldown,
        "workers_success_step": args.workers_success_step,
    }
    print(
        "[Metrics] selected="
        + ",".join(
            metric
            for metric in SUPPORTED_METRICS
            if metric in selected_metrics
        ),
        flush=True,
    )

    input_path = Path(args.input)
    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(f"{args.output}.checkpoint.json")

    record_cache_path: Optional[Path]
    if args.no_record_cache:
        record_cache_path = None
    elif args.record_cache:
        record_cache_path = Path(args.record_cache)
    else:
        record_cache_path = Path(f"{args.input}.record_cache.json")

    all_records = read_jsonl(input_path)

    profile_alignment_cache: Dict[str, Dict[str, Any]] = {}
    behavior_extraction_cache: Dict[str, Dict[str, Any]] = {}
    semantic_embedding_cache: Dict[str, Any] = {}
    cache_stats: Dict[str, int] = {
        "profile_alignment_hits": 0,
        "profile_alignment_misses": 0,
        "behavior_extraction_hits": 0,
        "behavior_extraction_misses": 0,
        "semantic_embedding_hits": 0,
        "semantic_embedding_misses": 0,
    }

    if record_cache_path is not None:
        cache_payload = load_record_cache(record_cache_path)
        loaded_profile_alignment = cache_payload.get("profile_alignment", {})
        loaded_behavior_extraction = cache_payload.get("behavior_extraction", {})
        if isinstance(loaded_profile_alignment, dict):
            profile_alignment_cache = loaded_profile_alignment
        if isinstance(loaded_behavior_extraction, dict):
            behavior_extraction_cache = loaded_behavior_extraction
        print(
            "[RecordCache] Loaded "
            f"profile_alignment={len(profile_alignment_cache)} "
            f"behavior_extraction={len(behavior_extraction_cache)} "
            f"from {record_cache_path}",
            flush=True,
        )
    else:
        print(
            "[RecordCache] Using in-memory caches only (not persisted): "
            "profile_alignment, behavior_extraction, semantic_embedding",
            flush=True,
        )

    report: Optional[Dict[str, Any]] = None
    try:
        run_range = tuple(args.run_range) if args.run_range else None
        if run_range is not None and run_range[0] != run_range[1]:
            if args.run_number or args.run_numbers:
                raise ValueError(
                    "When --run-range START END has START!=END (sweep mode), "
                    "do not combine with --run-number/--run-numbers."
                )
            start, end = run_range
            if start <= 0 or end <= 0:
                raise ValueError("--run-range values must be >= 1.")
            if start > end:
                raise ValueError(f"--run-range start must be <= end (got {start}>{end})")

            report = run_prefix_sweep(
                ks=list(range(start, end + 1)),
                all_records=all_records,
                input_path=input_path,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
                profile_alignment_cache=profile_alignment_cache,
                behavior_extraction_cache=behavior_extraction_cache,
                semantic_embedding_cache=semantic_embedding_cache,
                cache_stats=cache_stats,
                include_profile_alignment=include_profile_alignment,
                include_semantic_diversity=include_semantic_diversity,
                include_behavior_diversity=include_behavior_diversity,
                include_group_diversity=include_group_diversity,
                include_min_distance_diversity=include_min_distance_diversity,
                behavior_max_workers=args.workers,
                behavior_worker_tuning=behavior_worker_tuning,
            )
        else:
            requested_runs = resolve_requested_runs(
                run_number=args.run_number,
                run_numbers=args.run_numbers,
                run_range=run_range,
            )
            if requested_runs is None:
                records = all_records
                distribution, _ = derive_run_number_distributions(records)
                print(
                    "[RunFilter] No run filter provided. "
                    f"Evaluating all records ({len(records)}): {format_run_counts(distribution)}",
                    flush=True,
                )
            else:
                records = filter_for_requested_runs(
                    all_records=all_records,
                    requested_runs=requested_runs,
                )
            report = evaluate_selected_records(
                records=records,
                input_path=input_path,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                checkpoint_every=args.checkpoint_every,
                resume=args.resume,
                profile_alignment_cache=profile_alignment_cache,
                behavior_extraction_cache=behavior_extraction_cache,
                semantic_embedding_cache=semantic_embedding_cache,
                cache_stats=cache_stats,
                include_profile_alignment=include_profile_alignment,
                include_semantic_diversity=include_semantic_diversity,
                include_behavior_diversity=include_behavior_diversity,
                include_group_diversity=include_group_diversity,
                include_min_distance_diversity=include_min_distance_diversity,
                behavior_max_workers=args.workers,
                behavior_worker_tuning=behavior_worker_tuning,
            )
    except KeyboardInterrupt:
        print(
            f"Interrupted. Partial progress saved to checkpoint: {checkpoint_path}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130) from None
    finally:
        if (
            record_cache_path is not None
            and profile_alignment_cache is not None
            and behavior_extraction_cache is not None
        ):
            try:
                save_record_cache(
                    record_cache_path,
                    profile_alignment_cache=profile_alignment_cache,
                    behavior_extraction_cache=behavior_extraction_cache,
                )
                print(
                    "[RecordCache] Saved "
                    f"profile_alignment={len(profile_alignment_cache)} "
                    f"behavior_extraction={len(behavior_extraction_cache)} "
                    f"to {record_cache_path}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[RecordCache] Failed to save cache ({exc})",
                    file=sys.stderr,
                    flush=True,
                )

    if report is None:
        raise RuntimeError("Metric evaluation did not produce a report.")

    if report.get("mode") == "run_range_prefix_sweep":
        print(f"Wrote sweep manifest: {output_path}")
    elif report.get("num_records") is not None:
        print(f"Wrote metrics: {summarize_report_for_log(report)} (checkpoint={checkpoint_path})")
    else:
        print(f"Wrote output: {output_path}")

    if record_cache_path is not None:
        print(
            "[RecordCache] Reuse summary: "
            f"profile_alignment hits={cache_stats['profile_alignment_hits']} "
            f"misses={cache_stats['profile_alignment_misses']}; "
            f"behavior_extraction hits={cache_stats['behavior_extraction_hits']} "
            f"misses={cache_stats['behavior_extraction_misses']}; "
            f"semantic_embedding hits={cache_stats['semantic_embedding_hits']} "
            f"misses={cache_stats['semantic_embedding_misses']}",
            flush=True,
        )
    else:
        print(
            "[Cache] Reuse summary: "
            f"profile_alignment hits={cache_stats['profile_alignment_hits']} "
            f"misses={cache_stats['profile_alignment_misses']}; "
            f"behavior_extraction hits={cache_stats['behavior_extraction_hits']} "
            f"misses={cache_stats['behavior_extraction_misses']}; "
            f"semantic_embedding hits={cache_stats['semantic_embedding_hits']} "
            f"misses={cache_stats['semantic_embedding_misses']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
