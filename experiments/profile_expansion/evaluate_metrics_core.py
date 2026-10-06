"""Core evaluation pipeline for profile-expansion metrics."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from experiments.profile_expansion.metrics.behavior_diversity import (
    METRIC_VARIANT as BEHAVIOR_METRIC_VARIANT,
    score_behavior_diversity,
)
from experiments.profile_expansion.metrics.common import mean
from experiments.profile_expansion.metrics.profile_alignment import ASPECT_KEYS, score_profile_alignment
from experiments.profile_expansion.metrics.simulation_diversity import (
    METRIC_VARIANT as SIMULATION_METRIC_VARIANT,
    score_simulation_diversity,
)

# Bump when metric semantics or checkpoint payload semantics change.
CHECKPOINT_SCHEMA_VERSION = 6


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
    return records


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_records_signature(records: List[Dict[str, Any]]) -> str:
    digest = hashlib.md5()
    for record in records:
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def conversation_hash(record: Dict[str, Any]) -> str:
    return hashlib.md5(
        json.dumps(record.get("transcript", []), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _scored(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Records the judge actually scored (a failed call has score None)."""
    return [item for item in items if item.get("score") is not None]


def _mean_or_none(values) -> Optional[float]:
    values = list(values)
    return mean(values) if values else None


def _alignment_ok(cached: Any) -> bool:
    """A cached alignment that is usable; failed judge calls are re-scored."""
    return isinstance(cached, dict) and cached.get("score") is not None


def build_profile_alignment_section(alignment_by_record: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for item in alignment_by_record:
        grouped[(item.get("model"), item.get("profile_id"))].append(item)

    by_model_profile = []
    for (model, profile_id), group in grouped.items():
        aspect_means: Dict[str, float] = {}
        for aspect in ASPECT_KEYS:
            values = [
                item["aspect_scores"].get(aspect)
                for item in group
                if item.get("aspect_scores", {}).get(aspect) is not None
            ]
            aspect_means[aspect] = mean(values) if values else 0.0

        by_model_profile.append(
            {
                "model": model,
                "profile_id": profile_id,
                "source_title": group[0].get("source_title"),
                "num_conversations": len(group),
                "num_scored": len(_scored(group)),
                "avg_score_1_to_5": _mean_or_none(item["avg_score_1_to_5"] for item in _scored(group)),
                "score": _mean_or_none(item["score"] for item in _scored(group)),
                "aspect_scores_1_to_5": aspect_means,
            }
        )

    by_model_profile.sort(key=lambda item: (str(item.get("model")), str(item.get("profile_id"))))

    grouped_cpm: Dict[Tuple[Any, Any, Any], List[Dict[str, Any]]] = defaultdict(list)
    for item in alignment_by_record:
        grouped_cpm[(item.get("model"), item.get("profile_id"), item.get("conversation_hash"))].append(item)

    by_conversation_model_profile = []
    for (model, profile_id, convo_hash), group in grouped_cpm.items():
        aspect_means: Dict[str, float] = {}
        for aspect in ASPECT_KEYS:
            values = [
                item["aspect_scores"].get(aspect)
                for item in group
                if item.get("aspect_scores", {}).get(aspect) is not None
            ]
            aspect_means[aspect] = mean(values) if values else 0.0

        by_conversation_model_profile.append(
            {
                "model": model,
                "profile_id": profile_id,
                "conversation_hash": convo_hash,
                "source_title": group[0].get("source_title"),
                "num_records": len(group),
                "num_scored": len(_scored(group)),
                "avg_score_1_to_5": _mean_or_none(item["avg_score_1_to_5"] for item in _scored(group)),
                "score": _mean_or_none(item["score"] for item in _scored(group)),
                "aspect_scores_1_to_5": aspect_means,
            }
        )

    by_conversation_model_profile.sort(
        key=lambda item: (str(item.get("model")), str(item.get("profile_id")), str(item.get("conversation_hash")))
    )

    return {
        "score": _mean_or_none(item["score"] for item in _scored(alignment_by_record)),
        "avg_score_1_to_5": _mean_or_none(item["avg_score_1_to_5"] for item in _scored(alignment_by_record)),
        "num_judge_failures": len(alignment_by_record) - len(_scored(alignment_by_record)),
        "records": alignment_by_record,
        "by_model_profile": by_model_profile,
        "by_conversation_model_profile": by_conversation_model_profile,
    }


def build_simulation_section(simulation_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    sorted_profiles = sorted(
        simulation_profiles,
        key=lambda item: (str(item.get("model")), str(item.get("profile_id"))),
    )
    score_adjusted = mean(item.get("simulation_diversity", 0.0) for item in sorted_profiles)
    score_raw = mean(
        item.get("simulation_diversity_raw", item.get("simulation_diversity", 0.0))
        for item in sorted_profiles
    )
    return {
        "metric": "simulation_diversity",
        "variant": SIMULATION_METRIC_VARIANT,
        "score": score_adjusted,
        "score_adjusted": score_adjusted,
        "score_raw": score_raw,
        "profiles": sorted_profiles,
    }


def build_behavior_section(behavior_profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    sorted_profiles = sorted(
        behavior_profiles,
        key=lambda item: (str(item.get("model")), str(item.get("profile_id"))),
    )
    scored_profiles = [item for item in sorted_profiles if item.get("scored", True)]
    return {
        "metric": "behavior_diversity",
        "variant": BEHAVIOR_METRIC_VARIANT,
        "score": mean(item.get("behavior_diversity", 0.0) for item in scored_profiles),
        "score_v1_legacy": mean(
            item.get("behavior_diversity_v1", item.get("behavior_diversity", 0.0))
            for item in scored_profiles
        ),
        "num_profiles_scored": len(scored_profiles),
        "num_profiles_total": len(sorted_profiles),
        "profiles": sorted_profiles,
    }


def build_report(
    *,
    num_records: int,
    alignment_by_record: List[Dict[str, Any]],
    behavior_profiles: List[Dict[str, Any]],
    simulation_profiles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "num_records": num_records,
        "profile_alignment": build_profile_alignment_section(alignment_by_record),
        "behavior_diversity": build_behavior_section(behavior_profiles),
        "simulation_diversity": build_simulation_section(simulation_profiles or []),
    }


def build_report_partial(
    *,
    num_records: int,
    include_profile_alignment: bool,
    include_behavior_diversity: bool,
    include_simulation_diversity: bool = True,
    alignment_by_record: List[Dict[str, Any]],
    behavior_profiles: List[Dict[str, Any]],
    simulation_profiles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"num_records": num_records}
    if include_profile_alignment:
        report["profile_alignment"] = build_profile_alignment_section(alignment_by_record)
    if include_behavior_diversity:
        report["behavior_diversity"] = build_behavior_section(behavior_profiles)
    if include_simulation_diversity:
        report["simulation_diversity"] = build_simulation_section(simulation_profiles or [])
    return report


def checkpoint_summary(
    *,
    alignment_by_record: List[Dict[str, Any]],
    behavior_profiles: List[Dict[str, Any]],
    simulation_profiles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    simulation_profiles = simulation_profiles or []
    return {
        "profile_alignment_norm": _mean_or_none(item["score"] for item in _scored(alignment_by_record)),
        "profile_alignment_avg_1_to_5": _mean_or_none(
            item["avg_score_1_to_5"] for item in _scored(alignment_by_record)
        ),
        "behavior_diversity": mean(
            item.get("behavior_diversity", 0.0)
            for item in behavior_profiles
            if item.get("scored", True)
        ),
        "simulation_diversity": mean(
            item.get("simulation_diversity", 0.0) for item in simulation_profiles
        ),
        "simulation_diversity_raw": mean(
            item.get("simulation_diversity_raw", item.get("simulation_diversity", 0.0))
            for item in simulation_profiles
        ),
    }


def write_checkpoint(
    checkpoint_path: Path,
    *,
    input_path: Path,
    output_path: Path,
    input_signature: str,
    num_records: int,
    stage: str,
    status: str,
    alignment_by_record: List[Dict[str, Any]],
    behavior_profiles: List[Dict[str, Any]],
    simulation_profiles: Optional[List[Dict[str, Any]]] = None,
    message: Optional[str] = None,
    final_report: Optional[Dict[str, Any]] = None,
) -> None:
    simulation_profiles = simulation_profiles or []
    payload: Dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "status": status,
        "stage": stage,
        "updated_at_utc": utc_now_iso(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_signature": input_signature,
        "num_records_total": num_records,
        "progress": {
            "profile_alignment_records_completed": len(alignment_by_record),
            "behavior_profiles_completed": len(behavior_profiles),
            "simulation_profiles_completed": len(simulation_profiles),
        },
        "summary": checkpoint_summary(
            alignment_by_record=alignment_by_record,
            behavior_profiles=behavior_profiles,
            simulation_profiles=simulation_profiles,
        ),
        "state": {
            "alignment_records": alignment_by_record,
            "behavior_profiles": behavior_profiles,
            "simulation_profiles": simulation_profiles,
        },
    }
    if message:
        payload["message"] = message
    if final_report is not None:
        payload["report"] = final_report
    write_json(checkpoint_path, payload)


def _as_dict_list(raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def normalize_alignment_resume(alignment_records: List[Dict[str, Any]], num_records: int) -> List[Dict[str, Any]]:
    by_index: Dict[int, Dict[str, Any]] = {}
    for item in alignment_records:
        idx = item.get("conversation_index")
        if isinstance(idx, int) and 0 <= idx < num_records:
            by_index[idx] = item

    normalized: List[Dict[str, Any]] = []
    next_idx = 0
    while next_idx in by_index:
        normalized.append(by_index[next_idx])
        next_idx += 1

    if len(normalized) < len(by_index):
        print(
            "[Checkpoint] Profile-alignment checkpoint has gaps; "
            "resuming from the longest contiguous prefix only.",
            flush=True,
        )
    return normalized


def normalize_profile_resume(profile_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for item in profile_items:
        key = (item.get("model"), item.get("profile_id"))
        by_key[key] = item
    return list(by_key.values())


def load_resume_state(
    *,
    checkpoint_path: Path,
    input_signature: str,
    num_records: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not checkpoint_path.exists():
        return [], [], []

    try:
        with checkpoint_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[Checkpoint] Failed to parse checkpoint ({exc}); ignoring resume state.", flush=True)
        return [], [], []

    if not isinstance(payload, dict):
        print("[Checkpoint] Invalid checkpoint format; ignoring resume state.", flush=True)
        return [], [], []

    schema_version = payload.get("schema_version")
    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        print(
            f"[Checkpoint] Unsupported schema_version={schema_version}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}. Ignoring.",
            flush=True,
        )
        return [], [], []

    if payload.get("input_signature") != input_signature:
        print("[Checkpoint] Input signature mismatch; starting from scratch.", flush=True)
        return [], [], []

    if payload.get("num_records_total") != num_records:
        print("[Checkpoint] Record count mismatch; starting from scratch.", flush=True)
        return [], [], []

    state = payload.get("state", {}) if isinstance(payload.get("state"), dict) else {}
    alignment_records = _as_dict_list(state.get("alignment_records"))
    behavior_profiles = _as_dict_list(state.get("behavior_profiles"))
    simulation_profiles = _as_dict_list(state.get("simulation_profiles"))

    if not state:
        report = payload.get("report", {}) if isinstance(payload.get("report"), dict) else {}
        profile_alignment = report.get("profile_alignment", {}) if isinstance(report.get("profile_alignment"), dict) else {}
        behavior = report.get("behavior_diversity", {}) if isinstance(report.get("behavior_diversity"), dict) else {}
        simulation = report.get("simulation_diversity", {}) if isinstance(report.get("simulation_diversity"), dict) else {}
        alignment_records = _as_dict_list(profile_alignment.get("records"))
        behavior_profiles = _as_dict_list(behavior.get("profiles"))
        simulation_profiles = _as_dict_list(simulation.get("profiles"))

    normalized_alignment = normalize_alignment_resume(alignment_records, num_records)
    normalized_behavior = normalize_profile_resume(behavior_profiles)
    normalized_simulation = normalize_profile_resume(simulation_profiles)
    return normalized_alignment, normalized_behavior, normalized_simulation


def evaluate(
    records: List[Dict[str, Any]],
    *,
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    input_signature: str,
    checkpoint_every: int,
    resume_alignment_by_record: Optional[List[Dict[str, Any]]] = None,
    resume_behavior_profiles: Optional[List[Dict[str, Any]]] = None,
    resume_simulation_profiles: Optional[List[Dict[str, Any]]] = None,
    profile_alignment_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    behavior_extraction_cache: Optional[Dict[str, Dict[str, Any]]] = None,
    embedding_cache: Optional[Dict[str, Any]] = None,
    cache_stats: Optional[Dict[str, int]] = None,
    record_cache_key_fn: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None,
    compute_profile_alignment: bool = True,
    compute_behavior_diversity: bool = True,
    compute_simulation_diversity: bool = True,
    behavior_max_workers: int = 1,
    behavior_worker_tuning: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    alignment_by_record: List[Dict[str, Any]] = list(resume_alignment_by_record or [])
    behavior_profiles: List[Dict[str, Any]] = list(resume_behavior_profiles or [])
    simulation_profiles: List[Dict[str, Any]] = list(resume_simulation_profiles or [])

    if compute_profile_alignment:
        current_stage = "profile_alignment"
    elif compute_behavior_diversity:
        current_stage = "behavior_diversity"
    elif compute_simulation_diversity:
        current_stage = "simulation_diversity"
    else:
        current_stage = "done"

    try:
        if compute_profile_alignment:
            start_idx = len(alignment_by_record)
            if start_idx:
                print(
                    f"[Checkpoint] Resuming profile alignment at conversation {start_idx + 1}/{len(records)}.",
                    flush=True,
                )

            # Optional parallel pre-scoring: profile alignment is a per-record LLM
            # call. When behavior_max_workers > 1, score the not-yet-done, uncached
            # records concurrently (adaptive AIMD concurrency) and fill the cache.
            # The sequential loop below then hits the cache for every record, so its
            # ordering and per-record checkpointing remain unchanged.
            if (
                behavior_max_workers
                and behavior_max_workers > 1
                and profile_alignment_cache is not None
                and record_cache_key_fn is not None
                and start_idx < len(records)
            ):
                import concurrent.futures
                from experiments.profile_expansion.metrics.behavior_diversity import _AdaptiveConcurrency

                pending: Dict[str, Dict[str, Any]] = {}
                for idx in range(start_idx, len(records)):
                    record = records[idx]
                    cache_key = record_cache_key_fn(record)
                    if not cache_key or cache_key in pending:
                        continue
                    if _alignment_ok(profile_alignment_cache.get(cache_key)):
                        continue
                    pending[cache_key] = record

                if pending:
                    _tuning = behavior_worker_tuning or {}
                    limiter = _AdaptiveConcurrency(
                        behavior_max_workers,
                        decrease_factor=_tuning.get("workers_decrease_factor", 0.5),
                        cooldown_seconds=_tuning.get("workers_cooldown_seconds", 3.0),
                        success_step=_tuning.get("workers_success_step", 0),
                    )
                    print(
                        f"[ProfileAlignment] adaptive pre-scoring records={len(pending)} "
                        f"max_workers={behavior_max_workers}",
                        flush=True,
                    )

                    def _score_one(cache_key: str, record: Dict[str, Any]):
                        limiter.acquire()
                        try:
                            result = score_profile_alignment(record, on_error=limiter.on_error)
                            limiter.on_success()
                            return cache_key, result
                        finally:
                            limiter.release()

                    with concurrent.futures.ThreadPoolExecutor(max_workers=behavior_max_workers) as executor:
                        futures = [
                            executor.submit(_score_one, cache_key, record)
                            for cache_key, record in pending.items()
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            cache_key, result = future.result()
                            if _alignment_ok(result):  # failures are re-scored below
                                profile_alignment_cache[cache_key] = result
                    print("[ProfileAlignment] adaptive pre-scoring complete", flush=True)

            for idx in range(start_idx, len(records)):
                record = records[idx]
                cache_key = record_cache_key_fn(record) if record_cache_key_fn else None
                cached_alignment = (
                    profile_alignment_cache.get(cache_key)
                    if cache_key is not None and profile_alignment_cache is not None
                    else None
                )
                if _alignment_ok(cached_alignment):
                    alignment = cached_alignment
                    if cache_stats is not None:
                        cache_stats["profile_alignment_hits"] = cache_stats.get("profile_alignment_hits", 0) + 1
                else:
                    alignment = score_profile_alignment(record)
                    if cache_key is not None and profile_alignment_cache is not None and _alignment_ok(alignment):
                        profile_alignment_cache[cache_key] = alignment
                    if cache_stats is not None:
                        cache_stats["profile_alignment_misses"] = cache_stats.get("profile_alignment_misses", 0) + 1

                convo_hash = conversation_hash(record)
                print(
                    f"[ProfileAlignment] conversation={idx + 1}/{len(records)} "
                    f"profile_id={record.get('profile_id')} model={record.get('model')} "
                    + (f"avg_1_to_5={alignment['avg_score_1_to_5']:.3f}" if _alignment_ok(alignment)
                       else f"JUDGE FAILED ({alignment.get('parse_error') or 'no scores'})"),
                    flush=True,
                )
                alignment_by_record.append(
                    {
                        "conversation_index": idx,
                        "profile_id": record.get("profile_id"),
                        "source_title": record.get("source_title"),
                        "short_patient_profile": record.get("short_patient_profile"),
                        "model": record.get("model"),
                        "run_number": record.get("_derived_run_number"),
                        "run_index": record.get("run_index"),
                        "conversation_hash": convo_hash,
                        **alignment,
                    }
                )

                should_write_checkpoint = (
                    checkpoint_every == 1 or (idx + 1) % checkpoint_every == 0 or (idx + 1) == len(records)
                )
                if should_write_checkpoint:
                    write_checkpoint(
                        checkpoint_path,
                        input_path=input_path,
                        output_path=output_path,
                        input_signature=input_signature,
                        num_records=len(records),
                        stage=current_stage,
                        status="running",
                        alignment_by_record=alignment_by_record,
                        behavior_profiles=behavior_profiles,
                        simulation_profiles=simulation_profiles,
                    )
                    print(
                        f"[Checkpoint] stage={current_stage} saved "
                        f"alignment={len(alignment_by_record)}/{len(records)}",
                        flush=True,
                    )

        if compute_behavior_diversity:
            current_stage = "behavior_diversity"
            write_checkpoint(
                checkpoint_path,
                input_path=input_path,
                output_path=output_path,
                input_signature=input_signature,
                num_records=len(records),
                stage=current_stage,
                status="running",
                alignment_by_record=alignment_by_record,
                behavior_profiles=behavior_profiles,
                simulation_profiles=simulation_profiles,
            )

            def on_behavior_profile_scored(profile_score: Dict[str, Any], completed: int, total: int) -> None:
                behavior_profiles.append(profile_score)
                write_checkpoint(
                    checkpoint_path,
                    input_path=input_path,
                    output_path=output_path,
                    input_signature=input_signature,
                    num_records=len(records),
                    stage=current_stage,
                    status="running",
                    alignment_by_record=alignment_by_record,
                    behavior_profiles=behavior_profiles,
                    simulation_profiles=simulation_profiles,
                )
                print(
                    f"[Checkpoint] stage={current_stage} saved profiles={completed}/{total}",
                    flush=True,
                )

            behavior = score_behavior_diversity(
                records,
                existing_profiles=list(behavior_profiles),
                on_profile_scored=on_behavior_profile_scored,
                extraction_cache=behavior_extraction_cache,
                extraction_cache_key_fn=record_cache_key_fn,
                extraction_cache_stats=cache_stats,
                max_workers=behavior_max_workers,
                **(behavior_worker_tuning or {}),
            )
            behavior_profiles = normalize_profile_resume(_as_dict_list(behavior.get("profiles")))

        if compute_simulation_diversity:
            current_stage = "simulation_diversity"
            write_checkpoint(
                checkpoint_path,
                input_path=input_path,
                output_path=output_path,
                input_signature=input_signature,
                num_records=len(records),
                stage=current_stage,
                status="running",
                alignment_by_record=alignment_by_record,
                behavior_profiles=behavior_profiles,
                simulation_profiles=simulation_profiles,
            )

            def on_simulation_profile_scored(profile_score: Dict[str, Any], completed: int, total: int) -> None:
                simulation_profiles.append(profile_score)
                write_checkpoint(
                    checkpoint_path,
                    input_path=input_path,
                    output_path=output_path,
                    input_signature=input_signature,
                    num_records=len(records),
                    stage=current_stage,
                    status="running",
                    alignment_by_record=alignment_by_record,
                    behavior_profiles=behavior_profiles,
                    simulation_profiles=simulation_profiles,
                )
                print(
                    f"[Checkpoint] stage={current_stage} saved profiles={completed}/{total}",
                    flush=True,
                )

            simulation = score_simulation_diversity(
                records,
                existing_profiles=list(simulation_profiles),
                on_profile_scored=on_simulation_profile_scored,
                embedding_cache=embedding_cache,
                embedding_cache_stats=cache_stats,
            )
            simulation_profiles = normalize_profile_resume(_as_dict_list(simulation.get("profiles")))

        report = build_report_partial(
            num_records=len(records),
            include_profile_alignment=compute_profile_alignment,
            include_behavior_diversity=compute_behavior_diversity,
            include_simulation_diversity=compute_simulation_diversity,
            alignment_by_record=alignment_by_record,
            behavior_profiles=behavior_profiles,
            simulation_profiles=simulation_profiles,
        )
        write_checkpoint(
            checkpoint_path,
            input_path=input_path,
            output_path=output_path,
            input_signature=input_signature,
            num_records=len(records),
            stage="done",
            status="completed",
            alignment_by_record=alignment_by_record,
            behavior_profiles=behavior_profiles,
            simulation_profiles=simulation_profiles,
            final_report=report,
        )
        return report

    except KeyboardInterrupt:
        write_checkpoint(
            checkpoint_path,
            input_path=input_path,
            output_path=output_path,
            input_signature=input_signature,
            num_records=len(records),
            stage=current_stage,
            status="interrupted",
            alignment_by_record=alignment_by_record,
            behavior_profiles=behavior_profiles,
            simulation_profiles=simulation_profiles,
            message=f"Interrupted during stage={current_stage}.",
        )
        raise
    except Exception as exc:
        write_checkpoint(
            checkpoint_path,
            input_path=input_path,
            output_path=output_path,
            input_signature=input_signature,
            num_records=len(records),
            stage=current_stage,
            status="failed",
            alignment_by_record=alignment_by_record,
            behavior_profiles=behavior_profiles,
            simulation_profiles=simulation_profiles,
            message=f"{type(exc).__name__}: {exc}",
        )
        raise


__all__ = [
    "build_report_partial",
    "build_behavior_section",
    "build_simulation_section",
    "build_profile_alignment_section",
    "compute_records_signature",
    "evaluate",
    "load_resume_state",
    "read_jsonl",
    "write_json",
]
