#!/usr/bin/env python3
"""Run fixed-agenda profile-expansion experiments.

This file is intentionally small. It only handles CLI parsing, JSONL I/O, and
experiment orchestration. Model behavior lives in ``patient_models.py`` and the
topic-specific interview process lives in ``interview_process.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Heuristic: does this exception look like an API rate-limit / 429?"""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "ratelimit" in name
        or "429" in text
        or "rate limit" in text
        or "too many requests" in text
    )


class _AsyncAIMD:
    """Asyncio AIMD concurrency limiter for Azure-API patient conversations.

    Starts at ``max_workers`` concurrent conversations. On a rate-limit signal it
    multiplicatively decreases the allowed concurrency (debouncing bursts so a
    cluster of 429s from already in-flight calls counts once); while healthy it
    additively ramps back up toward ``max_workers``.
    """

    def __init__(
        self,
        max_workers: int,
        *,
        min_workers: int = 1,
        decrease_factor: float = 0.5,
        cooldown_seconds: float = 3.0,
        success_step: int = 0,
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.min_workers = max(1, min(int(min_workers), self.max_workers))
        self.limit = float(self.max_workers)
        self.in_use = 0
        self.decrease_factor = decrease_factor
        self.cooldown_seconds = cooldown_seconds
        self.success_step = success_step if success_step > 0 else self.max_workers
        self._success_since_increase = 0
        self._last_decrease = 0.0
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._cond:
            while self.in_use >= int(self.limit):
                await self._cond.wait()
            self.in_use += 1

    async def release(self) -> None:
        async with self._cond:
            self.in_use = max(0, self.in_use - 1)
            self._cond.notify_all()

    async def on_success(self) -> None:
        async with self._cond:
            if self.limit >= self.max_workers:
                return
            self._success_since_increase += 1
            if self._success_since_increase >= self.success_step:
                self._success_since_increase = 0
                new_limit = min(float(self.max_workers), self.limit + 1.0)
                if new_limit != self.limit:
                    self.limit = new_limit
                    print(f"[Workers] concurrency increase -> {int(self.limit)}", flush=True)
                    self._cond.notify_all()

    async def on_error(self, exc: BaseException) -> None:
        if not _is_rate_limit_error(exc):
            return
        now = time.time()
        async with self._cond:
            # Debounce: a batch of in-flight calls can all 429 at once; treat
            # rate-limit signals within the cooldown window as a single event.
            if now - self._last_decrease < self.cooldown_seconds:
                return
            self._last_decrease = now
            self._success_since_increase = 0
            new_limit = max(float(self.min_workers), float(int(self.limit * self.decrease_factor)))
            if new_limit != self.limit:
                self.limit = new_limit
                print(f"[Workers] rate-limit hit; concurrency decrease -> {int(self.limit)}", flush=True)

from angel_common.paths import REPO_ROOT

PACKAGE_ROOT = REPO_ROOT  # repository root; run scripts with `python -m` from here

from experiments.profile_expansion.interview_process import (
    TopicInterviewController,
    build_agenda_therapist,
)
from experiments.profile_expansion.patient_models import (
    EVALUATION_MODEL_CHOICES,
    build_evaluation_patient_model,
)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


def select_records(path: Path, start: int, limit: Optional[int]) -> List[Dict[str, Any]]:
    rows = list(iter_jsonl(path))
    selected = rows[start:]
    if limit is not None:
        selected = selected[:limit]
    return selected


def merge_profile_items(
    interview_item: Dict[str, Any],
    model_item: Dict[str, Any],
) -> Dict[str, Any]:
    merged = dict(model_item)
    merged.update(interview_item)
    return merged


def profile_key(profile_id: Any, source_title: Any, short_profile: Any) -> Tuple[str, str, str]:
    profile_id_key = "" if profile_id is None else str(profile_id)
    source_title_key = str(source_title or "").strip()
    short_profile_hash = hashlib.md5(str(short_profile or "").encode("utf-8")).hexdigest()
    return (profile_id_key, source_title_key, short_profile_hash)


def resolve_short_profile_for_key(item: Dict[str, Any]) -> str:
    short_profile = item.get("short_patient_profile")
    if isinstance(short_profile, str) and short_profile.strip():
        return short_profile.strip()

    patient_payload = item.get("patient_processed_result")
    if isinstance(patient_payload, dict):
        complaint = patient_payload.get("complaints")
        if isinstance(complaint, str) and complaint.strip():
            return complaint.strip()

    return ""


def completed_runs_from_output(
    output_path: Path,
    *,
    model_label: str,
    num_runs: int,
) -> Dict[Tuple[str, str, str], Set[int]]:
    if not output_path.exists():
        return {}

    indexed_runs_by_profile: Dict[Tuple[str, str, str], Set[int]] = defaultdict(set)
    legacy_count_by_profile: Dict[Tuple[str, str, str], int] = defaultdict(int)
    parse_errors = 0
    ignored_other_model = 0

    with output_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue

            if row.get("model") != model_label:
                ignored_other_model += 1
                continue

            key = profile_key(
                row.get("profile_id"),
                row.get("source_title"),
                row.get("short_patient_profile"),
            )

            run_index = row.get("run_index")
            if isinstance(run_index, int) and run_index >= 0:
                indexed_runs_by_profile[key].add(run_index)
            else:
                # Backward compatibility: old rows did not include run_index.
                legacy_count_by_profile[key] += 1

    completed_by_profile: Dict[Tuple[str, str, str], Set[int]] = {}
    for key in set(indexed_runs_by_profile) | set(legacy_count_by_profile):
        done = {idx for idx in indexed_runs_by_profile.get(key, set()) if idx < num_runs}
        legacy_count = legacy_count_by_profile.get(key, 0)
        if legacy_count:
            for idx in range(num_runs):
                if idx not in done:
                    done.add(idx)
                    legacy_count -= 1
                    if legacy_count <= 0:
                        break
        if done:
            completed_by_profile[key] = done

    if parse_errors:
        print(
            f"[Resume] skipped {parse_errors} malformed JSONL lines in existing output: {output_path}",
            flush=True,
        )
    if ignored_other_model:
        print(
            f"[Resume] ignored {ignored_other_model} rows with other model labels in existing output.",
            flush=True,
        )

    return completed_by_profile


async def _run_one_conversation(
    *,
    args: argparse.Namespace,
    controller: "TopicInterviewController",
    limiter: _AsyncAIMD,
    write_lock: asyncio.Lock,
    counter: Dict[str, int],
    merged_item: Dict[str, Any],
    input_index: int,
    run_index: int,
    output_path: Path,
) -> None:
    max_retries = 6
    attempt = 0
    while True:
        await limiter.acquire()
        try:
            # Fresh model per task: API patient models hold a per-profile system
            # prompt, so concurrent tasks must not share a mutable instance.
            patient_model = build_evaluation_patient_model(
                model_name=args.model,
                profile_item=merged_item,
                current_model=None,
                verbose=False,
            )
            record = await controller.run(
                patient_model=patient_model,
                profile_item=merged_item,
                model_label=args.model,
                run_index=run_index,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - log and (maybe) retry
            await limiter.on_error(exc)
            await limiter.release()
            if _is_rate_limit_error(exc) and attempt < max_retries:
                attempt += 1
                await asyncio.sleep(min(2 ** attempt, 30))
                continue
            print(
                f"SKIP profile_id={merged_item.get('id')} run={run_index} "
                f"model={args.model}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return
        await limiter.on_success()
        await limiter.release()

        record["run_index"] = run_index
        record["input_index"] = input_index
        async with write_lock:
            append_jsonl(output_path, record)
            counter["done"] += 1
            print(
                f"[{counter['done']}/{counter.get('total', 0)}] wrote "
                f"profile_id={record.get('profile_id')} run={run_index} model={args.model}",
                flush=True,
            )
        return


async def run_experiment_concurrent(
    *,
    args: argparse.Namespace,
    controller: "TopicInterviewController",
    profiles: List[Dict[str, Any]],
    model_init_rows: List[Dict[str, Any]],
    output_path: Path,
    existing_completed_by_profile: Dict[Tuple[str, str, str], Set[int]],
) -> None:
    limiter = _AsyncAIMD(
        args.workers,
        decrease_factor=args.workers_decrease_factor,
        cooldown_seconds=args.workers_cooldown,
        success_step=args.workers_success_step,
    )
    write_lock = asyncio.Lock()
    counter: Dict[str, int] = {"done": 0}

    tasks = []
    for offset, profile_item in enumerate(profiles):
        input_index = args.start + offset
        model_item = model_init_rows[offset]
        merged_item = merge_profile_items(profile_item, model_item)
        case_key = profile_key(
            merged_item.get("id"),
            merged_item.get("source_title"),
            resolve_short_profile_for_key(merged_item),
        )
        done_runs = set(existing_completed_by_profile.get(case_key, set()))
        for run_index in range(args.num_runs):
            if run_index in done_runs:
                continue
            tasks.append(
                _run_one_conversation(
                    args=args,
                    controller=controller,
                    limiter=limiter,
                    write_lock=write_lock,
                    counter=counter,
                    merged_item=merged_item,
                    input_index=input_index,
                    run_index=run_index,
                    output_path=output_path,
                )
            )

    total = len(tasks)
    counter["total"] = total
    print(
        f"[Workers] launching {total} pending conversations with adaptive "
        f"concurrency (start={args.workers}, decrease_factor={args.workers_decrease_factor}, "
        f"cooldown={args.workers_cooldown}s, success_step={args.workers_success_step or args.workers})",
        flush=True,
    )
    await asyncio.gather(*tasks)
    print(f"[Workers] done: {counter['done']}/{total} conversations written.", flush=True)


async def run_experiment(args: argparse.Namespace) -> None:
    profiles = select_records(Path(args.input), args.start, args.limit)
    output_path = Path(args.output)
    existing_completed_by_profile: Dict[Tuple[str, str, str], Set[int]] = {}
    if args.resume:
        existing_completed_by_profile = completed_runs_from_output(
            output_path,
            model_label=args.model,
            num_runs=args.num_runs,
        )

    print(
        f"[Init] model={args.model} runs={args.num_runs} input_rows={len(profiles)} resume={args.resume}",
        flush=True,
    )

    if args.model == "angel":
        has_payload = bool(profiles and isinstance(profiles[0].get("angel_processed_result"), dict))
        model_init_rows = profiles
        if not has_payload and args.verbose:
            print(
                "[Init] angel payload missing in input rows; using two-stage Angel initialization "
                "(short profile -> long profile -> Angel model).",
                flush=True,
            )
    else:
        has_payload = bool(profiles and isinstance(profiles[0].get("patient_processed_result"), dict))
        model_init_rows = profiles
        if not has_payload and args.verbose:
            print(
                f"[Init] patient payload missing in input rows; using short-profile fallback initialization for {args.model}.",
                flush=True,
            )

    if args.verbose:
        print(
            f"[Init] loaded interview_rows={len(profiles)} model_init_rows={len(model_init_rows)}",
            flush=True,
        )

    if len(model_init_rows) < len(profiles):
        raise ValueError(
            f"Not enough model-init rows: need {len(profiles)}, found {len(model_init_rows)} "
            "in selected input rows."
        )

    controller = TopicInterviewController(
        therapist=build_agenda_therapist("azure"),
        transition_judge=None,
        transition_policy="therapist",
        max_therapist_turns_per_topic=args.max_therapist_turns_per_topic,
        verbose=args.verbose,
    )

    if args.workers and args.workers > 1:
        if args.model in {"angel", "eeyore", "one_stage"}:
            print(
                f"[Workers] WARNING: --workers={args.workers} with local GPU model "
                f"'{args.model}' is not recommended (single-process GPU contention / OOM). "
                "Prefer multi-GPU data sharding via CUDA_VISIBLE_DEVICES + --start/--limit. "
                "Proceeding anyway.",
                flush=True,
            )
        await run_experiment_concurrent(
            args=args,
            controller=controller,
            profiles=profiles,
            model_init_rows=model_init_rows,
            output_path=output_path,
            existing_completed_by_profile=existing_completed_by_profile,
        )
        return

    total = len(profiles) * args.num_runs
    completed = 0
    patient_model: Any = None
    for offset, profile_item in enumerate(profiles):
        input_index = args.start + offset
        model_item = model_init_rows[offset]
        merged_item = merge_profile_items(profile_item, model_item)
        case_key = profile_key(
            merged_item.get("id"),
            merged_item.get("source_title"),
            resolve_short_profile_for_key(merged_item),
        )
        completed_runs_for_case = set(existing_completed_by_profile.get(case_key, set()))
        completed += len(completed_runs_for_case)

        left_title = (profile_item.get("source_title") or "").strip()
        right_title = (model_item.get("source_title") or "").strip()
        if left_title and right_title and left_title != right_title:
            raise ValueError(
                "Input/eval row mismatch after slicing. "
                f"input[{input_index}] title={left_title!r}, eval[{input_index}] title={right_title!r}"
            )

        if args.verbose:
            print(
                f"[Case {offset + 1}/{len(profiles)}] "
                f"input_index={input_index} profile_id={merged_item.get('id')} "
                f"title={merged_item.get('source_title', '')} "
                f"precompleted_runs={len(completed_runs_for_case)}",
                flush=True,
            )
        elif completed_runs_for_case:
            print(
                f"[Case {offset + 1}/{len(profiles)}] resume precompleted_runs={len(completed_runs_for_case)} "
                f"profile_id={merged_item.get('id')}",
                flush=True,
            )

        for run_index in range(args.num_runs):
            if run_index in completed_runs_for_case:
                if args.verbose:
                    print(
                        f"[Case {offset + 1}/{len(profiles)}][Run {run_index + 1}/{args.num_runs}] "
                        "skip (already in output)",
                        flush=True,
                    )
                continue

            patient_model = build_evaluation_patient_model(
                model_name=args.model,
                profile_item=merged_item,
                current_model=patient_model,
                verbose=args.verbose,
            )
            if args.verbose:
                print(
                    f"[Case {offset + 1}/{len(profiles)}][Run {run_index + 1}/{args.num_runs}] "
                    f"patient instance={type(patient_model).__name__} ready",
                    flush=True,
                )
            try:
                record = await controller.run(
                    patient_model=patient_model,
                    profile_item=merged_item,
                    model_label=args.model,
                    run_index=run_index,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:
                print(
                    f"[Case {offset + 1}/{len(profiles)}][Run {run_index + 1}/{args.num_runs}] "
                    f"SKIP profile_id={merged_item.get('id')} run={run_index} model={args.model}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            record["run_index"] = run_index
            record["input_index"] = input_index
            append_jsonl(output_path, record)

            completed += 1
            print(
                f"[{completed}/{total}] wrote profile_id={record.get('profile_id')} "
                f"run={run_index} model={args.model}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate profile-expansion patient conversations with a fixed topic agenda."
    )
    parser.add_argument(
        "--input",
        default=str(layout.SHORT_PROFILES),
        help="Input JSONL containing short_patient_profile fields.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path. Records are appended.")

    parser.add_argument(
        "--model",
        required=True,
        choices=list(EVALUATION_MODEL_CHOICES),
        help="AI patient model for agenda experiments.",
    )

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--num-runs", type=int, default=1, help="Runs per profile for diversity measurement.")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--max-therapist-turns-per-topic", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Max concurrent conversations. >1 enables adaptive AIMD concurrency "
            "(intended for Azure-API patient models: patient_psi, roleplay_doh). "
            "Starts at this many in-flight, backs off on 429, ramps back up. Default: 1."
        ),
    )
    parser.add_argument(
        "--workers-decrease-factor",
        type=float,
        default=0.5,
        help="AIMD multiplicative decrease factor on a rate-limit hit (0<f<1). Default: 0.5.",
    )
    parser.add_argument(
        "--workers-cooldown",
        type=float,
        default=3.0,
        help="AIMD cooldown seconds; 429s within this window of the last decrease count once. Default: 3.0.",
    )
    parser.add_argument(
        "--workers-success-step",
        type=int,
        default=0,
        help="Successful conversations per +1 concurrency. 0 = auto (= --workers). Default: 0.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output by skipping already-finished profile runs.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed debug logs.")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_experiment(parse_args()))


if __name__ == "__main__":
    main()
