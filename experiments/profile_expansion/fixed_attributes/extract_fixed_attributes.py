#!/usr/bin/env python3
"""Extract fixed profile attributes from short profiles using GPT.

This script reads input JSONL rows, extracts schema-aligned fixed attributes
from each profile text, and writes the same rows with two extra keys:
- fixed_attribute
- fixed_attribute_number
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from angel_common.paths import REPO_ROOT

PACKAGE_ROOT = REPO_ROOT  # repository root; run scripts with `python -m` from here

from angel_common.llm import getOutput
from experiments.profile_expansion.metrics.behavior_diversity import TOPIC_ATTRIBUTE_SCHEMA


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid JSON object on {path}:{line_no}: expected dict row.")
            yield payload


def _canonical_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON: {text[:240]}")


def _find_by_canonical_key(payload: Dict[str, Any], target: str) -> Any:
    if target in payload:
        return payload[target]
    target_norm = _canonical_key(target)
    for key, value in payload.items():
        if _canonical_key(str(key)) == target_norm:
            return value
    return None


def _normalize_values(raw: Any) -> Set[str]:
    values: List[str] = []
    if raw is None:
        return set()
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, (list, tuple, set)):
        values = [str(item) for item in raw if item is not None]
    else:
        values = [str(raw)]

    cleaned: Set[str] = set()
    for value in values:
        value = re.sub(r"\s+", " ", value).strip().lower()
        value = value.strip(".,;:!?")
        if value:
            cleaned.add(value)
    return cleaned


def _resolve_short_profile(record: Dict[str, Any]) -> str:
    short_profile = record.get("short_patient_profile")
    if isinstance(short_profile, str) and short_profile.strip():
        return short_profile.strip()

    patient_psi_profile = record.get("patient_psi_profile")
    if isinstance(patient_psi_profile, dict):
        for key in ("history", "complaints", "summary"):
            value = patient_psi_profile.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if isinstance(patient_psi_profile, str) and patient_psi_profile.strip():
        return patient_psi_profile.strip()

    return ""


def _profile_cache_key(record: Dict[str, Any]) -> Tuple[str, str, str]:
    profile_id = str(record.get("id") or record.get("profile_id") or "")
    source_title = str(record.get("source_title") or "").strip()
    short_hash = hashlib.md5(_resolve_short_profile(record).encode("utf-8")).hexdigest()
    return (profile_id, source_title, short_hash)


def _empty_fixed_attribute_payload() -> Dict[str, Dict[str, List[str]]]:
    return {
        topic_key: {attr: [] for attr in spec["attributes"]}
        for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items()
    }


def _serialize_extracted(
    extracted: Dict[str, Dict[str, Set[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    return {
        topic_key: {attr: sorted(values) for attr, values in topic_values.items()}
        for topic_key, topic_values in extracted.items()
    }


def _count_nonempty_attributes(
    fixed_attribute: Dict[str, Dict[str, List[str]]],
) -> int:
    return sum(
        1
        for topic_values in fixed_attribute.values()
        for values in topic_values.values()
        if isinstance(values, list) and len(values) > 0
    )


def _build_prompt(short_profile: str) -> str:
    schema_text = json.dumps(
        {
            topic_key: {
                "goal": spec["goal"],
                "attributes": spec["attributes"],
            }
            for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items()
        },
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are extracting fixed patient profile attributes from a short clinical profile.

Task:
- Use only the profile text evidence.
- Extract stable/fixed profile facts (not hypothetical details).
- Do not invent unsupported details.
- Use short phrase values.

Short patient profile:
{short_profile}

Topic schema:
{schema_text}

Return valid JSON only with this structure:
{{
  "topics": {{
    "<topic_key>": {{
      "<attribute_key>": ["value1", "value2"]
    }}
  }}
}}

Rules:
- Include every topic key and every attribute key from the schema.
- Use [] when a fixed attribute is not supported by the profile.
""".strip()


def extract_fixed_attributes(
    short_profile: str,
    *,
    max_completion_tokens: int,
    max_retries: int,
) -> Tuple[Dict[str, Dict[str, List[str]]], int, str]:
    extracted: Dict[str, Dict[str, Set[str]]] = {
        topic_key: {attr: set() for attr in spec["attributes"]}
        for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items()
    }
    if not short_profile.strip():
        payload = _serialize_extracted(extracted)
        return payload, 0, "empty_short_profile"

    prompt = _build_prompt(short_profile)
    parse_error = ""

    try:
        output_text = getOutput(
            prompt,
            context=None,
            max_completion_tokens=max_completion_tokens,
            tag=0,
            max_retries=max_retries,
        )
        parsed = _extract_json(output_text)
        topics_block = parsed.get("topics") if isinstance(parsed, dict) else None
        if not isinstance(topics_block, dict):
            topics_block = parsed if isinstance(parsed, dict) else {}

        for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items():
            topic_obj = _find_by_canonical_key(topics_block, topic_key)
            if not isinstance(topic_obj, dict):
                continue
            for attr in spec["attributes"]:
                raw = _find_by_canonical_key(topic_obj, attr)
                extracted[topic_key][attr] = _normalize_values(raw)
    except Exception as exc:
        parse_error = repr(exc)

    payload = _serialize_extracted(extracted)
    fixed_attribute_number = _count_nonempty_attributes(payload)
    return payload, fixed_attribute_number, parse_error


def load_resume_cache(path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    if not path.exists():
        return {}
    cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if "fixed_attribute" not in row or "fixed_attribute_number" not in row:
            continue
        cache[_profile_cache_key(row)] = row
    return cache


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract fixed attributes from profile JSONL.")
    parser.add_argument(
        "--input",
        default=str(layout.SHORT_PROFILES_V3),
        help="Input JSONL with short_patient_profile.",
    )
    parser.add_argument(
        "--output",
        default=str(layout.INPUT_DIR / "selected_50_short_patient_profiles_v3.fixed_attributes.jsonl"),
        help="Output JSONL path with fixed_attribute fields added.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start offset in input rows.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to process.")
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=1800,
        help="max_completion_tokens passed to getOutput.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retry count passed to getOutput.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse rows from existing output when profile key matches.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0 when provided")

    input_path = Path(args.input)
    output_path = Path(args.output)
    rows = list(iter_jsonl(input_path))
    selected = rows[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    resume_cache: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if args.resume:
        resume_cache = load_resume_cache(output_path)

    print(
        f"[Init] input={input_path} selected_rows={len(selected)} output={output_path} "
        f"resume={args.resume} resume_hits_available={len(resume_cache)}",
        flush=True,
    )

    processed: List[Dict[str, Any]] = []
    num_reused = 0
    num_api_calls = 0
    num_parse_errors = 0

    for idx, row in enumerate(selected, 1):
        out_row = dict(row)
        key = _profile_cache_key(row)
        cached = resume_cache.get(key)
        if cached is not None:
            out_row["fixed_attribute"] = cached.get("fixed_attribute", _empty_fixed_attribute_payload())
            try:
                out_row["fixed_attribute_number"] = int(cached.get("fixed_attribute_number", 0))
            except Exception:
                out_row["fixed_attribute_number"] = _count_nonempty_attributes(
                    out_row["fixed_attribute"]
                    if isinstance(out_row["fixed_attribute"], dict)
                    else _empty_fixed_attribute_payload()
                )
            processed.append(out_row)
            num_reused += 1
            if args.verbose:
                print(
                    f"[Row {idx}/{len(selected)}] reused profile_id={row.get('id')}",
                    flush=True,
                )
            continue

        short_profile = _resolve_short_profile(row)
        fixed_attribute, fixed_attribute_number, parse_error = extract_fixed_attributes(
            short_profile,
            max_completion_tokens=args.max_completion_tokens,
            max_retries=args.max_retries,
        )
        out_row["fixed_attribute"] = fixed_attribute
        out_row["fixed_attribute_number"] = fixed_attribute_number
        processed.append(out_row)
        num_api_calls += 1

        if parse_error:
            num_parse_errors += 1
            print(
                f"[Row {idx}/{len(selected)}] profile_id={row.get('id')} parse_error={parse_error}",
                flush=True,
            )
        elif args.verbose:
            print(
                f"[Row {idx}/{len(selected)}] profile_id={row.get('id')} "
                f"fixed_attribute_number={fixed_attribute_number}",
                flush=True,
            )

    write_jsonl(output_path, processed)
    print(
        f"[Done] wrote_rows={len(processed)} reused={num_reused} api_calls={num_api_calls} "
        f"parse_errors={num_parse_errors} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
