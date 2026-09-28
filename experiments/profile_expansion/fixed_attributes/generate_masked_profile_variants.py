#!/usr/bin/env python3
"""Generate fixed-attribute-masked variants of short patient profiles (Claude-backed).

Pipeline per input row:
1. Resolve the short patient profile text.
2. Extract its fixed attributes using TOPIC_ATTRIBUTE_SCHEMA via Claude. Let N be
   the number of non-empty fixed attributes.
3. For every target attribute count k in 1..N, sample up to ``--samples-per-count``
   random attribute combinations to KEEP (masking the remaining N-k). Selection
   is seeded for reproducibility.
4. For every masked combination, call Claude to rewrite the profile text so the
   masked facts are removed while every kept fact and the prose fluency are kept.
5. Save the original profile plus all variants to a per-profile JSON file under
   ``--output-dir`` (one file per original profile, named ``<profile_id>.json``).

Both extraction and rewrite use the Anthropic API (ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL),
defaulting to ``claude-opus-4-7``. Work is parallelized two ways: across profiles
(``--profile-workers``) and across rewrites within a profile (``--workers``).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Any, Dict, List, Optional, Tuple

from angel_common.paths import REPO_ROOT

PACKAGE_ROOT = REPO_ROOT  # repository root; run scripts with `python -m` from here

import anthropic

from experiments.profile_expansion.metrics.behavior_diversity import TOPIC_ATTRIBUTE_SCHEMA
from experiments.profile_expansion.fixed_attributes.extract_fixed_attributes import (
    _build_prompt,
    _count_nonempty_attributes,
    _empty_fixed_attribute_payload,
    _extract_json,
    _find_by_canonical_key,
    _normalize_values,
    _resolve_short_profile,
    _serialize_extracted,
    iter_jsonl,
)

# An attribute is identified by (topic_key, attribute_key) and carries its values.
AttributeRef = Tuple[str, str]

# Shared Anthropic client. httpx-backed clients are safe to share across threads.
_CLIENT: Optional[anthropic.Anthropic] = None


def init_client() -> anthropic.Anthropic:
    global _CLIENT
    if _CLIENT is None:
        # max_retries=0: our own loop owns retry/backoff so the adaptive limiter
        # observes every rate-limit event. Reads ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL.
        from angel_common.llm import anthropic_client

        client = anthropic_client()
        _CLIENT = client.with_options(max_retries=0, timeout=120.0) if hasattr(client, "with_options") else client
    return _CLIENT


class AdaptiveLimiter:
    """AIMD concurrency limiter shared across all worker threads.

    Starts allowing ``initial`` concurrent Claude calls. On a rate-limit signal it
    multiplicatively halves the effective limit (down to ``min_limit``); after a
    run of successes it additively grows back toward ``max_limit``. This auto-tunes
    the real concurrency below the thread-pool ceiling when the API pushes back.
    """

    def __init__(self, initial: int, min_limit: int = 1, max_limit: Optional[int] = None):
        self._cond = threading.Condition()
        self._limit = max(1, initial)
        self._min = max(1, min_limit)
        self._max = max(self._limit, max_limit or initial)
        self._active = 0
        self._successes = 0

    def acquire(self) -> None:
        with self._cond:
            while self._active >= self._limit:
                self._cond.wait()
            self._active += 1

    def release(self) -> None:
        with self._cond:
            self._active -= 1
            self._cond.notify()

    def on_rate_limit(self) -> int:
        with self._cond:
            self._limit = max(self._min, self._limit // 2)
            self._successes = 0
            return self._limit

    def on_success(self) -> None:
        with self._cond:
            if self._limit >= self._max:
                return
            self._successes += 1
            # Grow one slot per ~2*limit successes (gentle additive increase).
            if self._successes >= self._limit * 2:
                self._limit += 1
                self._successes = 0
                self._cond.notify()

    @property
    def limit(self) -> int:
        with self._cond:
            return self._limit


# Set in main() once worker counts are known.
_LIMITER: Optional[AdaptiveLimiter] = None


def _is_rate_limit(exc: Exception) -> bool:
    if isinstance(exc, anthropic.RateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    return status in (429, 529, 503)


def _retry_after_seconds(exc: Exception, default: float) -> float:
    """Honor a Retry-After header if the API provides one, else use ``default``."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw:
            try:
                return max(default, float(raw))
            except (TypeError, ValueError):
                pass
    return default


def claude_generate(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    max_retries: int,
    base_sleep: float = 2.0,
    max_sleep: float = 60.0,
) -> Tuple[str, str]:
    """Call the Claude messages API. Returns (text, error).

    Retries on API exceptions and empty completions. Rate-limit errors trigger the
    shared adaptive limiter (shrinking concurrency) and exponential backoff that
    honors Retry-After. ``text`` is "" on persistent failure; ``error`` carries the
    last failure reason ("" on success).
    """
    client = init_client()
    limiter = _LIMITER
    last_error = "empty_output"
    # Rate-limit retries are cheap to grant generously; they don't consume "real"
    # attempts so a throttled call still gets its content attempts.
    attempt = 0
    rate_limit_hits = 0
    max_rate_limit_hits = 8
    while attempt < max(1, max_retries):
        if limiter is not None:
            limiter.acquire()
        try:
            message = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:  # network / rate-limit / API errors
            last_error = repr(exc)
            if _is_rate_limit(exc) and rate_limit_hits < max_rate_limit_hits:
                rate_limit_hits += 1
                new_limit = limiter.on_rate_limit() if limiter is not None else None
                wait = _retry_after_seconds(exc, min(max_sleep, base_sleep * (2 ** rate_limit_hits)))
                print(
                    f"[RateLimit] hit#{rate_limit_hits} limit->{new_limit} sleep={wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)
                continue  # does not count toward content attempts
            attempt += 1
            time.sleep(base_sleep)
            continue
        finally:
            if limiter is not None:
                limiter.release()

        if limiter is not None:
            limiter.on_success()
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()
        if text:
            return text, ""
        attempt += 1
        time.sleep(base_sleep)
    return "", last_error


# -------------------------------------------------------------------------
# Fixed-attribute extraction (Claude). Reuses the schema-aware prompt/parsing
# helpers from extract_fixed_attributes.py; only the model call differs.
# -------------------------------------------------------------------------
def extract_fixed_attributes_claude(
    short_profile: str,
    *,
    model: str,
    max_tokens: int,
    max_retries: int,
) -> Tuple[Dict[str, Dict[str, List[str]]], int, str]:
    extracted: Dict[str, Dict[str, set]] = {
        topic_key: {attr: set() for attr in spec["attributes"]}
        for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items()
    }
    if not short_profile.strip():
        return _serialize_extracted(extracted), 0, "empty_short_profile"

    prompt = _build_prompt(short_profile)
    output_text, parse_error = claude_generate(
        prompt, model=model, max_tokens=max_tokens, max_retries=max_retries
    )
    if output_text:
        try:
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
            parse_error = ""
        except Exception as exc:
            parse_error = repr(exc)

    payload = _serialize_extracted(extracted)
    return payload, _count_nonempty_attributes(payload), parse_error


def _present_attributes(
    fixed_attribute: Dict[str, Dict[str, List[str]]],
) -> List[Tuple[AttributeRef, List[str]]]:
    """Return non-empty (topic_key, attribute_key) refs with their values."""
    present: List[Tuple[AttributeRef, List[str]]] = []
    for topic_key, spec in TOPIC_ATTRIBUTE_SCHEMA.items():
        topic_values = fixed_attribute.get(topic_key, {})
        for attr in spec["attributes"]:
            values = topic_values.get(attr)
            if isinstance(values, list) and len(values) > 0:
                present.append(((topic_key, attr), list(values)))
    return present


def _masked_fixed_attribute(
    kept: List[Tuple[AttributeRef, List[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    """Build a schema-shaped fixed_attribute dict containing only kept values."""
    payload = _empty_fixed_attribute_payload()
    for (topic_key, attr), values in kept:
        payload[topic_key][attr] = list(values)
    return payload


def _format_attribute_lines(
    items: List[Tuple[AttributeRef, List[str]]],
) -> str:
    lines = []
    for (topic_key, attr), values in items:
        joined = ", ".join(values)
        lines.append(f"- {topic_key}.{attr}: {joined}")
    return "\n".join(lines) if lines else "- (none)"


def _sample_combinations(
    n: int,
    k: int,
    samples_per_count: int,
    rng: random.Random,
) -> List[Tuple[int, ...]]:
    """Sample up to ``samples_per_count`` distinct size-k index subsets of range(n).

    If C(n, k) <= samples_per_count, every combination is returned.
    """
    total = math.comb(n, k)
    if total <= samples_per_count:
        return [tuple(c) for c in combinations(range(n), k)]

    seen: set = set()
    subsets: List[Tuple[int, ...]] = []
    # Bounded attempts so we never loop forever on degenerate cases.
    max_attempts = samples_per_count * 50
    attempts = 0
    while len(subsets) < samples_per_count and attempts < max_attempts:
        attempts += 1
        pick = tuple(sorted(rng.sample(range(n), k)))
        if pick not in seen:
            seen.add(pick)
            subsets.append(pick)
    return subsets


def _build_rewrite_prompt(
    original_text: str,
    masked: List[Tuple[AttributeRef, List[str]]],
    kept: List[Tuple[AttributeRef, List[str]]],
) -> str:
    return f"""
You are editing a short clinical patient profile. Your only job is to REMOVE a
specific set of facts while leaving every other fact intact.

Original patient profile:
{original_text}

REMOVE these facts (after editing they must no longer be stated or inferable):
{_format_attribute_lines(masked)}

KEEP these facts (they must remain clearly present and unchanged in meaning):
{_format_attribute_lines(kept)}

Rules:
- Remove ONLY the facts listed under REMOVE. Do not drop, weaken, or alter any
  other information, including the KEEP facts.
- Do not add new facts or invent any detail that is not in the original.
- After removal, rephrase as needed so the profile still reads as fluent, natural
  clinical prose. Preserve the original tone and style.
- Return ONLY the rewritten profile text. No JSON, no headings, no commentary.
""".strip()


def _rewrite_profile(
    original_text: str,
    masked: List[Tuple[AttributeRef, List[str]]],
    kept: List[Tuple[AttributeRef, List[str]]],
    *,
    model: str,
    max_tokens: int,
    max_retries: int,
) -> Tuple[str, str]:
    """Rewrite the profile removing masked facts. Returns (text, error)."""
    if not masked:
        # Nothing to mask (k == N): the variant equals the original profile.
        return original_text, ""
    prompt = _build_rewrite_prompt(original_text, masked, kept)
    text, error = claude_generate(
        prompt, model=model, max_tokens=max_tokens, max_retries=max_retries
    )
    if text:
        return text, ""
    return original_text, error


def build_profile_variants(
    row: Dict[str, Any],
    *,
    model: str,
    samples_per_count: int,
    seed: int,
    extract_max_tokens: int,
    rewrite_max_tokens: int,
    max_retries: int,
    workers: int,
    verbose: bool,
) -> Dict[str, Any]:
    profile_id = row.get("id")
    short_profile = _resolve_short_profile(row)

    fixed_attribute, fixed_attribute_number, extract_error = extract_fixed_attributes_claude(
        short_profile,
        model=model,
        max_tokens=extract_max_tokens,
        max_retries=max_retries,
    )
    present = _present_attributes(fixed_attribute)
    n = len(present)

    # Per-profile RNG so sampling is reproducible regardless of processing order.
    rng = random.Random(f"{seed}-{profile_id}")

    # Enumerate all variant jobs first (sampling stays sequential/seeded), then
    # run the LLM rewrites concurrently since each Claude call is blocking I/O.
    jobs: List[Dict[str, Any]] = []
    for k in range(1, n + 1):
        index_subsets = _sample_combinations(n, k, samples_per_count, rng)
        for sample_idx, kept_indices in enumerate(index_subsets):
            kept_set = set(kept_indices)
            kept = [present[i] for i in kept_indices]
            masked = [present[i] for i in range(n) if i not in kept_set]
            jobs.append({"k": k, "sample_idx": sample_idx, "kept": kept, "masked": masked})

    def _run_job(job: Dict[str, Any]) -> Dict[str, Any]:
        rewritten_text, rewrite_error = _rewrite_profile(
            short_profile,
            job["masked"],
            job["kept"],
            model=model,
            max_tokens=rewrite_max_tokens,
            max_retries=max_retries,
        )
        masked_payload = _masked_fixed_attribute(job["kept"])
        return {
            "variant_id": f"{profile_id}_k{job['k']}_s{job['sample_idx']}",
            "target_attribute_number": job["k"],
            "fixed_attribute_number": _count_nonempty_attributes(masked_payload),
            "kept_attributes": [list(ref) for ref, _ in job["kept"]],
            "masked_attributes": [list(ref) for ref, _ in job["masked"]],
            "fixed_attribute": masked_payload,
            "short_patient_profile": rewritten_text,
            "rewrite_error": rewrite_error,
        }

    if jobs and workers > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            variants = list(pool.map(_run_job, jobs))
    else:
        variants = [_run_job(job) for job in jobs]

    if verbose:
        n_err = sum(1 for v in variants if v["rewrite_error"])
        print(
            f"  profile_id={profile_id} N={n} variants={len(variants)} "
            f"rewrite_errors={n_err} extract_error={extract_error or '-'}",
            flush=True,
        )

    return {
        "profile_id": profile_id,
        "source_title": row.get("source_title"),
        "name": row.get("name"),
        "age": row.get("age"),
        "gender": row.get("gender"),
        "diagnosis_hint": row.get("diagnosis_hint"),
        "model": model,
        "original_short_patient_profile": short_profile,
        "original_fixed_attribute": fixed_attribute,
        "original_fixed_attribute_number": fixed_attribute_number,
        "extract_error": extract_error,
        "samples_per_count": samples_per_count,
        "seed": seed,
        "num_variants": len(variants),
        "variants": variants,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed-attribute-masked variants of short patient profiles."
    )
    parser.add_argument(
        "--input",
        default=str(layout.SHORT_PROFILES_V3),
        help="Input JSONL with short_patient_profile.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(layout.FIXATTR_PROFILES_DIR),
        help="Directory for per-profile variant JSON files.",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-7",
        help="Anthropic model for extraction and rewrite.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start offset in input rows.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to process.")
    parser.add_argument(
        "--samples-per-count",
        type=int,
        default=3,
        help="Max random attribute combinations to sample for each target count k.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed (per-profile derived).")
    parser.add_argument(
        "--extract-max-tokens",
        type=int,
        default=4000,
        help="max_tokens for fixed-attribute extraction. The schema has 141 "
        "attributes across 14 topics, so the full JSON needs ~1.5k+ tokens.",
    )
    parser.add_argument(
        "--rewrite-max-tokens",
        type=int,
        default=1000,
        help="max_tokens for each masking rewrite.",
    )
    parser.add_argument("--max-retries", type=int, default=3, help="Retry count for Claude calls.")
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Concurrent rewrite calls within a single profile.",
    )
    parser.add_argument(
        "--profile-workers",
        type=int,
        default=4,
        help="Number of profiles processed concurrently. Total concurrency is "
        "roughly profile-workers * workers.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip profiles whose output JSON file already exists.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0 when provided")
    if args.samples_per_count <= 0:
        raise ValueError("--samples-per-count must be > 0")
    if args.workers <= 0 or args.profile_workers <= 0:
        raise ValueError("--workers and --profile-workers must be > 0")

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    rows = list(iter_jsonl(input_path))
    selected = rows[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]

    init_client()  # create the shared client before spawning threads

    # The thread pools cap threads at profile_workers * workers; the adaptive
    # limiter governs how many of those may actually be in-flight at once and
    # backs off automatically when the API returns rate-limit errors.
    global _LIMITER
    max_concurrency = args.profile_workers * args.workers
    _LIMITER = AdaptiveLimiter(initial=max_concurrency, min_limit=1, max_limit=max_concurrency)

    print(
        f"[Init] input={input_path} selected_rows={len(selected)} output_dir={output_dir} "
        f"model={args.model} samples_per_count={args.samples_per_count} seed={args.seed} "
        f"profile_workers={args.profile_workers} workers={args.workers} "
        f"max_concurrency={max_concurrency} resume={args.resume}",
        flush=True,
    )

    total = len(selected)

    def process_one(indexed_row: Tuple[int, Dict[str, Any]]) -> Tuple[str, Any, str]:
        idx, row = indexed_row
        profile_id = row.get("id")
        out_path = output_dir / f"{profile_id}.json"
        if args.resume and out_path.exists():
            return ("skipped", profile_id, "")
        try:
            payload = build_profile_variants(
                row,
                model=args.model,
                samples_per_count=args.samples_per_count,
                seed=args.seed,
                extract_max_tokens=args.extract_max_tokens,
                rewrite_max_tokens=args.rewrite_max_tokens,
                max_retries=args.max_retries,
                workers=args.workers,
                verbose=args.verbose,
            )
        except Exception as exc:
            return ("failed", profile_id, f"{type(exc).__name__}: {exc}")
        write_json(out_path, payload)
        return ("done", profile_id, str(out_path))

    num_written = 0
    num_skipped = 0
    num_failed = 0
    done_count = 0

    with ThreadPoolExecutor(max_workers=args.profile_workers) as pool:
        for status, profile_id, detail in pool.map(process_one, enumerate(selected, 1)):
            done_count += 1
            if status == "done":
                num_written += 1
                print(
                    f"[{done_count}/{total}] wrote profile_id={profile_id} -> {detail}",
                    flush=True,
                )
            elif status == "skipped":
                num_skipped += 1
                if args.verbose:
                    print(
                        f"[{done_count}/{total}] skip existing profile_id={profile_id}",
                        flush=True,
                    )
            else:  # failed
                num_failed += 1
                print(
                    f"[{done_count}/{total}] FAILED profile_id={profile_id}: {detail}",
                    flush=True,
                )

    print(
        f"[Done] written={num_written} skipped={num_skipped} failed={num_failed} "
        f"output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
