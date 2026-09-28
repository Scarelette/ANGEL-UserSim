from __future__ import annotations

"""
Codebook LLM judge: score model responses against Codebook.txt with Claude.

Run from the repository root:
    python -m experiments.safety_exp.codebook_llm_judge --input-jsonl <responses.jsonl> ...
"""

import argparse
from pathlib import Path

from angel_common.env import get_env
from angel_common.llm import anthropic_client
from angel_common.paths import OUTPUTS_DIR

from . import codebook_judge_core as _core
from .codebook_judge_core import *  # noqa: F401,F403


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge scorer for safety_exp Codebook using Claude."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(default_model_responses_dir()),
        help="Directory containing model response JSONL files.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help=(
            "Single JSONL file of responses (e.g., output from query_models.py). "
            "When provided, --input-dir/--file-glob are ignored."
        ),
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default="*.jsonl",
        help="Glob used within --input-dir to find response files.",
    )
    parser.add_argument(
        "--codebook-path",
        type=str,
        default=str(default_codebook_path()),
        help="Path to Codebook.txt.",
    )
    parser.add_argument(
        "--prompt-template-path",
        type=str,
        default=str(default_prompt_template_path()),
        help=(
            "Path to prompt template used for LLM judging. "
            "Template must include {{PROMPT_NAME}}, {{CONTEXT_LEVEL}}, {{RESPONSE_TEXT}}, "
            "{{SINGLE_PROMPT_FOCUS}}, {{CODEBOOK_PROCEDURAL_NOTES}}, "
            "{{CODEBOOK_UNIVERSAL_RUBRICS}}, {{CODEBOOK_SINGLE_PROMPT_RUBRICS}}."
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=str(OUTPUTS_DIR / "safety_exp" / "codebook_judge_results.jsonl"),
        help="Path to append per-response raw judge output JSONL.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(OUTPUTS_DIR / "safety_exp" / "codebook_judge_long.csv"),
        help="Path to write long-form CSV with study-compatible columns.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-6",
        help="Claude model name.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Anthropic API key. Prefer the ANTHROPIC_API_KEY env var (keys on the command line end up in shell history).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1800,
        help="Max tokens per judge call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retry attempts per response.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.0,
        help="Sleep between retries.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index into parsed response records.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of records to process after --offset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs instead of resume/append mode.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files and print counts only; do not call Claude.",
    )
    args = parser.parse_args()

    if args.max_tokens <= 0:
        parser.error("--max-tokens must be > 0.")
    if args.max_retries < 1:
        parser.error("--max-retries must be >= 1.")
    if args.sleep_seconds < 0:
        parser.error("--sleep-seconds must be >= 0.")
    if args.offset < 0:
        parser.error("--offset must be >= 0.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0 when provided.")

    return args


def main() -> None:
    args = parse_args()

    input_jsonl = Path(args.input_jsonl) if args.input_jsonl else None
    input_dir = Path(args.input_dir)
    codebook_path = Path(args.codebook_path)
    prompt_template_path = Path(args.prompt_template_path)
    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)

    if input_jsonl is not None:
        if not input_jsonl.exists():
            raise FileNotFoundError(f"Input JSONL file does not exist: {input_jsonl}")
    else:
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not codebook_path.exists():
        raise FileNotFoundError(f"Codebook file does not exist: {codebook_path}")
    if not prompt_template_path.exists():
        raise FileNotFoundError(f"Prompt template does not exist: {prompt_template_path}")

    if input_jsonl is not None:
        records = parse_response_jsonl_file(input_jsonl)
    else:
        records = load_records(input_dir=input_dir, file_glob=args.file_glob)
    records = iter_slice(records, args.offset, args.limit)

    print(f"Parsed records: {len(records)}")
    if not records:
        print("No records found. Exiting.")
        return

    unknown_meta = [
        r for r in records if r.model_id is None or r.context_id is None or r.prompt_id is None
    ]
    if unknown_meta:
        print(f"Warning: {len(unknown_meta)} records have unknown model/context/prompt id mapping.")

    if args.dry_run:
        return

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        if output_jsonl.exists():
            output_jsonl.unlink()
        existing_keys: set[str] = set()
        next_row_id = 1
    else:
        existing_keys, max_existing_row_id = load_existing_keys(output_jsonl)
        next_row_id = max_existing_row_id + 1

    codebook_sections = load_codebook_sections(codebook_path)
    prompt_template_text = load_prompt_template(prompt_template_path)
    # Direct Anthropic API by default; set ANTHROPIC_BASE_URL to route through a
    # compatible endpoint (e.g. Azure AI Foundry).
    client = anthropic_client(api_key=args.api_key or get_env("ANTHROPIC_API_KEY"))

    processed = 0
    skipped = 0
    failed = 0

    for idx, record in enumerate(records, start=1):
        if record.stable_key in existing_keys:
            skipped += 1
            continue

        prompt = build_judge_prompt(
            codebook_sections=codebook_sections,
            prompt_template=prompt_template_text,
            prompt_name=record.prompt_raw,
            context_level=record.context_raw,
            response_text=record.response_text,
        )
        print(
            f"[{idx}/{len(records)}] "
            f"{record.source_file}#{record.source_file_index} "
            f"prompt='{record.prompt_raw}' context='{record.context_raw}'"
        )

        try:
            judgment = call_claude_with_retries(
                client=client,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_retries=args.max_retries,
                sleep_seconds=args.sleep_seconds,
            )
        except Exception as exc:
            failed += 1
            failure_payload = {
                "stable_key": record.stable_key,
                "source_file": record.source_file,
                "source_file_index": record.source_file_index,
                "model_raw": record.model_raw,
                "context_raw": record.context_raw,
                "prompt_raw": record.prompt_raw,
                "row_id": None,
                "error": str(exc),
            }
            write_jsonl_line(output_jsonl, failure_payload)
            print(f"  -> FAILED: {exc}")
            continue

        row_id = next_row_id
        next_row_id += 1

        csv_row = build_csv_row(row_id=row_id, record=record, judgment=judgment)

        payload = {
            "stable_key": record.stable_key,
            "source_file": record.source_file,
            "source_file_index": record.source_file_index,
            "row_id": row_id,
            "model_raw": record.model_raw,
            "context_raw": record.context_raw,
            "prompt_raw": record.prompt_raw,
            "model_id": record.model_id,
            "context_id": record.context_id,
            "prompt_id": record.prompt_id,
            "response_text": record.response_text,
            "judgment": judgment,
            "csv_row": csv_row,
        }
        write_jsonl_line(output_jsonl, payload)
        processed += 1

    # Rebuild CSV from all successful JSONL records so resume mode keeps CSV consistent.
    csv_rows_all = load_csv_rows_from_jsonl(output_jsonl)
    write_csv_rows(output_csv, csv_rows_all)

    print(
        "Done. "
        f"processed={processed}, skipped={skipped}, failed={failed}, "
        f"jsonl={output_jsonl}, csv={output_csv}"
    )


__all__ = ["main", "parse_args", *_core.__all__]


if __name__ == "__main__":
    main()
