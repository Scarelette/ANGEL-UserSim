"""Persistent per-record cache helpers for evaluate_metrics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

RECORD_CACHE_SCHEMA_VERSION = 1


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(path)


def record_cache_key(record: Dict[str, Any]) -> str:
    payload = {
        "model": record.get("model"),
        "profile_id": record.get("profile_id"),
        "source_title": record.get("source_title"),
        "short_patient_profile": record.get("short_patient_profile"),
        "transcript": record.get("transcript", []),
    }
    return hashlib.md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _empty_record_cache_payload() -> Dict[str, Any]:
    return {
        "schema_version": RECORD_CACHE_SCHEMA_VERSION,
        "updated_at_utc": _utc_now_iso(),
        "profile_alignment": {},
        "behavior_extraction": {},
    }


def load_record_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _empty_record_cache_payload()

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"[RecordCache] Failed to parse cache ({exc}); starting empty.", flush=True)
        return _empty_record_cache_payload()

    if not isinstance(payload, dict):
        print("[RecordCache] Invalid cache format; starting empty.", flush=True)
        return _empty_record_cache_payload()

    schema_version = payload.get("schema_version")
    if schema_version != RECORD_CACHE_SCHEMA_VERSION:
        print(
            f"[RecordCache] Unsupported schema_version={schema_version}; "
            f"expected {RECORD_CACHE_SCHEMA_VERSION}. Starting empty.",
            flush=True,
        )
        return _empty_record_cache_payload()

    alignment = payload.get("profile_alignment")
    behavior = payload.get("behavior_extraction")
    if not isinstance(alignment, dict):
        alignment = {}
    if not isinstance(behavior, dict):
        behavior = {}

    return {
        "schema_version": RECORD_CACHE_SCHEMA_VERSION,
        "updated_at_utc": _utc_now_iso(),
        "profile_alignment": {
            str(key): value
            for key, value in alignment.items()
            if isinstance(key, str) and isinstance(value, dict)
        },
        "behavior_extraction": {
            str(key): value
            for key, value in behavior.items()
            if isinstance(key, str) and isinstance(value, dict)
        },
    }


def save_record_cache(
    path: Path,
    *,
    profile_alignment_cache: Dict[str, Dict[str, Any]],
    behavior_extraction_cache: Dict[str, Dict[str, Any]],
) -> None:
    _write_json(
        path,
        {
            "schema_version": RECORD_CACHE_SCHEMA_VERSION,
            "updated_at_utc": _utc_now_iso(),
            "profile_alignment": profile_alignment_cache,
            "behavior_extraction": behavior_extraction_cache,
        },
    )


__all__ = [
    "load_record_cache",
    "record_cache_key",
    "save_record_cache",
]
