#!/usr/bin/env python3
"""Run fixed-attribute variant experiments over expanded profiles.

This mirrors ``run_agenda_experiment.py`` but drives generation from the
per-profile variant files in ``$ANGEL_OUTPUT_DIR/profile_expansion/fixed_attributes/profiles/*.json`` instead of a
flat JSONL of short profiles.

Selection / generation logic
----------------------------
1. Only profiles whose ``original_fixed_attribute_number`` is ``>= --min-original``
   (default 22) are eligible. Smaller ones are skipped.
2. For an eligible profile we walk ``fixed_attribute_number`` from 1 up to
   ``--max-fixed-attr`` (default 22). Each ``fixed_attribute_number`` has several
   variant *samples* (``..._k<K>_s0``, ``_s1``, ``_s2``).
     * We feed a variant's ``short_patient_profile`` into the interview pipeline
       and try to generate one transcript for that ``fixed_attribute_number``.
     * If a sample fails, we fall back to the next sample that shares the same
       ``fixed_attribute_number``.
     * If every sample for the current ``fixed_attribute_number`` fails (i.e. the
       next variant would belong to a different ``fixed_attribute_number``), we
       abandon the whole profile and move on to the next eligible profile.
3. A profile only "counts" when all ``fixed_attribute_number`` values 1..N each
   produced exactly one transcript. We keep going through eligible profiles until
   ``--target-profiles`` (default 10) profiles are fully completed.

With the defaults this yields ``10 * 22 = 220`` transcripts. To keep the output
clean (only fully-completed profiles), records for a profile are buffered and
appended atomically once the profile completes; abandoned profiles write nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Dict, List, Optional, Tuple

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

DEFAULT_PROFILES_DIR = str(layout.FIXATTR_PROFILES_DIR)

_SAMPLE_RE = re.compile(r"_s(\d+)$")


def load_profile(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_profile_paths(profiles_dir: Path) -> List[Path]:
    """Return profile json paths sorted numerically by stem when possible."""

    def sort_key(p: Path) -> Tuple[int, str]:
        try:
            return (0, f"{int(p.stem):012d}")
        except ValueError:
            return (1, p.stem)

    return sorted(profiles_dir.glob("*.json"), key=sort_key)


def sample_index_of(variant: Dict[str, Any], fallback: int) -> int:
    variant_id = str(variant.get("variant_id") or "")
    match = _SAMPLE_RE.search(variant_id)
    if match:
        return int(match.group(1))
    return fallback


def variants_by_fixed_number(
    profile: Dict[str, Any],
    max_fixed_attr: int,
) -> Dict[int, List[Dict[str, Any]]]:
    """Group variants by ``fixed_attribute_number`` (1..max_fixed_attr).

    Samples within a group are ordered by their ``_s<idx>`` suffix.
    """
    grouped: Dict[int, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for list_pos, variant in enumerate(profile.get("variants", [])):
        fan = variant.get("fixed_attribute_number")
        if not isinstance(fan, int) or fan < 1 or fan > max_fixed_attr:
            continue
        grouped[fan].append((sample_index_of(variant, list_pos), variant))

    ordered: Dict[int, List[Dict[str, Any]]] = {}
    for fan, items in grouped.items():
        items.sort(key=lambda pair: pair[0])
        ordered[fan] = [variant for _, variant in items]
    return ordered


def build_variant_profile_item(
    profile: Dict[str, Any],
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the ``profile_item`` consumed by the interview pipeline."""
    short_profile = variant.get("short_patient_profile")
    base_id = profile.get("profile_id")
    return {
        # ``interview_process`` reads ``id`` for the output ``profile_id`` field.
        "id": base_id,
        "profile_id": base_id,
        "source_title": profile.get("source_title"),
        "short_patient_profile": short_profile,
    }


def transcript_is_usable(record: Dict[str, Any]) -> bool:
    transcript = record.get("transcript")
    if not isinstance(transcript, list) or not transcript:
        return False
    # At least one topic must contain a patient turn for the record to be useful.
    for topic in transcript:
        for turn in topic.get("turns", []) if isinstance(topic, dict) else []:
            if turn.get("role") == "patient" and str(turn.get("content") or "").strip():
                return True
    return False


def append_records(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def completed_profile_ids(output_path: Path, max_fixed_attr: int) -> set:
    """Return base profile_ids already fully completed in an existing output."""
    if not output_path.exists():
        return set()

    seen_fixed: Dict[Any, set] = defaultdict(set)
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row.get("base_profile_id", row.get("profile_id"))
            fan = row.get("fixed_attribute_number")
            if isinstance(fan, int):
                seen_fixed[pid].add(fan)

    needed = set(range(1, max_fixed_attr + 1))
    return {pid for pid, fans in seen_fixed.items() if needed.issubset(fans)}


async def generate_one_profile(
    *,
    controller: TopicInterviewController,
    profile: Dict[str, Any],
    args: argparse.Namespace,
    patient_model_holder: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Try to produce one transcript per fixed_attribute_number for a profile.

    Returns the list of completed records, or ``None`` if the profile was
    abandoned because some fixed_attribute_number had no usable sample.
    """
    base_id = profile.get("profile_id")
    original_fan = profile.get("original_fixed_attribute_number")
    grouped = variants_by_fixed_number(profile, args.max_fixed_attr)

    records: List[Dict[str, Any]] = []
    for fixed_attr in range(1, args.max_fixed_attr + 1):
        samples = grouped.get(fixed_attr, [])
        if not samples:
            print(
                f"[Profile {base_id}] ABANDON: no variant for "
                f"fixed_attribute_number={fixed_attr}",
                flush=True,
            )
            return None

        produced: Optional[Dict[str, Any]] = None
        for variant in samples:
            short_profile = variant.get("short_patient_profile")
            if not isinstance(short_profile, str) or not short_profile.strip():
                continue

            profile_item = build_variant_profile_item(profile, variant)
            patient_model_holder["model"] = build_evaluation_patient_model(
                model_name=args.model,
                profile_item=profile_item,
                current_model=patient_model_holder.get("model"),
                verbose=args.verbose,
            )
            try:
                record = await controller.run(
                    patient_model=patient_model_holder["model"],
                    profile_item=profile_item,
                    model_label=args.model,
                    run_index=0,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - log and fall back
                print(
                    f"[Profile {base_id}][k={fixed_attr}] "
                    f"FAIL variant={variant.get('variant_id')}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                continue

            if not transcript_is_usable(record):
                print(
                    f"[Profile {base_id}][k={fixed_attr}] "
                    f"FAIL variant={variant.get('variant_id')}: empty transcript",
                    flush=True,
                )
                continue

            record["base_profile_id"] = base_id
            record["variant_id"] = variant.get("variant_id")
            record["fixed_attribute_number"] = fixed_attr
            record["sample_index"] = sample_index_of(variant, -1)
            record["original_fixed_attribute_number"] = original_fan
            record["run_index"] = 0
            produced = record
            print(
                f"[Profile {base_id}][k={fixed_attr}] OK "
                f"variant={variant.get('variant_id')}",
                flush=True,
            )
            break

        if produced is None:
            # All samples for this fixed_attribute_number failed -> abandon profile.
            print(
                f"[Profile {base_id}] ABANDON: all samples failed at "
                f"fixed_attribute_number={fixed_attr}",
                flush=True,
            )
            return None

        records.append(produced)

    return records


async def run_experiment(args: argparse.Namespace) -> None:
    profiles_dir = Path(args.profiles_dir)
    output_path = Path(args.output)

    already_done: set = set()
    if args.resume:
        already_done = completed_profile_ids(output_path, args.max_fixed_attr)
        if already_done:
            print(
                f"[Resume] {len(already_done)} profile(s) already complete in output: "
                f"{sorted(already_done)}",
                flush=True,
            )

    controller = TopicInterviewController(
        therapist=build_agenda_therapist("azure"),
        transition_judge=None,
        transition_policy="therapist",
        max_therapist_turns_per_topic=args.max_therapist_turns_per_topic,
        verbose=args.verbose,
    )

    print(
        f"[Init] model={args.model} target_profiles={args.target_profiles} "
        f"max_fixed_attr={args.max_fixed_attr} min_original={args.min_original} "
        f"profiles_dir={profiles_dir}",
        flush=True,
    )

    completed = len(already_done)
    patient_model_holder: Dict[str, Any] = {"model": None}

    for path in iter_profile_paths(profiles_dir):
        if completed >= args.target_profiles:
            break

        try:
            profile = load_profile(path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[Skip] cannot read {path.name}: {exc}", flush=True)
            continue

        base_id = profile.get("profile_id")
        original_fan = profile.get("original_fixed_attribute_number")

        if not isinstance(original_fan, int) or original_fan < args.min_original:
            if args.verbose:
                print(
                    f"[Skip] profile {base_id} original_fixed_attribute_number="
                    f"{original_fan} < {args.min_original}",
                    flush=True,
                )
            continue

        if base_id in already_done:
            print(f"[Skip] profile {base_id} already complete (resume).", flush=True)
            continue

        print(
            f"[Profile {base_id}] start (original_fixed_attribute_number={original_fan}) "
            f"[{completed}/{args.target_profiles} complete]",
            flush=True,
        )

        records = await generate_one_profile(
            controller=controller,
            profile=profile,
            args=args,
            patient_model_holder=patient_model_holder,
        )

        if records is None:
            print(f"[Profile {base_id}] incomplete; nothing written.", flush=True)
            continue

        append_records(output_path, records)
        completed += 1
        print(
            f"[Profile {base_id}] COMPLETE: wrote {len(records)} transcripts "
            f"[{completed}/{args.target_profiles}]",
            flush=True,
        )

    print(
        f"[Done] completed_profiles={completed}/{args.target_profiles} "
        f"output={output_path}",
        flush=True,
    )
    if completed < args.target_profiles:
        print(
            "[Warn] ran out of eligible profiles before reaching the target.",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fixed-attribute variant transcripts (k=1..N) for profiles "
            "whose original_fixed_attribute_number is large enough."
        )
    )
    parser.add_argument(
        "--profiles-dir",
        default=DEFAULT_PROFILES_DIR,
        help="Directory of per-profile variant json files.",
    )
    parser.add_argument("--output", required=True, help="Output JSONL path. Records are appended.")
    parser.add_argument(
        "--model",
        required=True,
        choices=list(EVALUATION_MODEL_CHOICES),
        help="AI patient model for transcript generation.",
    )

    parser.add_argument("--target-profiles", type=int, default=10, help="Number of fully-completed profiles wanted.")
    parser.add_argument("--max-fixed-attr", type=int, default=22, help="Generate one transcript per k in 1..this value.")
    parser.add_argument("--min-original", type=int, default=22, help="Only use profiles with original_fixed_attribute_number >= this.")

    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--max-therapist-turns-per-topic", type=int, default=8)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip profiles already fully completed in the existing output.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable detailed debug logs.")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_experiment(parse_args()))


if __name__ == "__main__":
    main()
