"""Run-selection utilities for evaluate_metrics CLI."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple


def parse_int_csv(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        part = token.strip()
        if not part:
            raise ValueError(f"Invalid empty run number in --run-numbers={raw!r}")
        try:
            values.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid run number {part!r} in --run-numbers={raw!r}") from exc
    return values


def resolve_requested_runs(
    *,
    run_number: Optional[List[int]],
    run_numbers: Optional[str],
    run_range: Optional[Tuple[int, int]],
) -> Optional[Set[int]]:
    requested: Set[int] = set()

    if run_number:
        requested.update(run_number)

    if run_numbers:
        requested.update(parse_int_csv(run_numbers))

    if run_range:
        start, end = run_range
        if start > end:
            raise ValueError(f"--run-range start must be <= end (got {start}>{end})")
        # Prefix semantics for --run-range:
        # - run_range K K  => include first K runs per profile (runs 1..K)
        # - run_range A B  => include first B runs per profile (runs 1..B)
        #
        # This matches "run number == how many runs are incorporated" behavior.
        requested.update(range(1, end + 1))

    if not requested:
        return None

    non_positive = sorted(value for value in requested if value <= 0)
    if non_positive:
        raise ValueError(
            "Run numbers are 1-based and must be >= 1. "
            f"Received invalid values: {non_positive}"
        )
    return requested


def format_run_counts(run_counts: Dict[int, int]) -> str:
    if not run_counts:
        return "(none)"
    return ", ".join(f"run_{run_number}:{count}" for run_number, count in sorted(run_counts.items()))


def derive_run_number_distributions(
    records: List[Dict[str, Any]],
) -> Tuple[Dict[int, int], List[int]]:
    per_profile_counter: Dict[Tuple[Any, Any], int] = defaultdict(int)
    derived_numbers: List[int] = []
    distribution: Dict[int, int] = defaultdict(int)

    for record in records:
        key = (record.get("model"), record.get("profile_id"))
        per_profile_counter[key] += 1
        run_number = per_profile_counter[key]
        derived_numbers.append(run_number)
        distribution[run_number] += 1

    return dict(distribution), derived_numbers


def filter_records_by_runs(
    records: List[Dict[str, Any]],
    requested_runs: Set[int],
) -> Tuple[List[Dict[str, Any]], Dict[int, int], Dict[int, int]]:
    all_distribution, derived_numbers = derive_run_number_distributions(records)
    filtered: List[Dict[str, Any]] = []
    selected_distribution: Dict[int, int] = defaultdict(int)

    for record, run_number in zip(records, derived_numbers):
        if run_number not in requested_runs:
            continue
        enriched = dict(record)
        enriched["_derived_run_number"] = run_number
        filtered.append(enriched)
        selected_distribution[run_number] += 1

    return filtered, dict(selected_distribution), all_distribution


__all__ = [
    "derive_run_number_distributions",
    "filter_records_by_runs",
    "format_run_counts",
    "resolve_requested_runs",
]
