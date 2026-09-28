import argparse
import glob
import json
import os
from typing import Any, Dict, List


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON in {path}:{line_no}: {exc}") from exc
    return rows


def normalize_case_key(item: Dict[str, Any]) -> str:
    item_id = item.get("id")
    if item_id is None:
        raise ValueError("Each row must contain an 'id' field.")
    return str(item_id)


def merge_profiles(input_paths: List[str]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for input_path in input_paths:
        for item in read_jsonl(input_path):
            case_key = normalize_case_key(item)
            short_profile = item.get("short_profile")
            long_profile = item.get("long_profile_description")

            if not isinstance(short_profile, str) or not short_profile.strip():
                raise ValueError(f"Missing or empty short_profile for case id={case_key} in {input_path}")
            if not isinstance(long_profile, str) or not long_profile.strip():
                raise ValueError(
                    f"Missing or empty long_profile_description for case id={case_key} in {input_path}"
                )

            if case_key not in grouped:
                grouped[case_key] = {
                    "original_id": item.get("id"),
                    "source_title": item.get("source_title"),
                    "short_profile": short_profile.strip(),
                    "generated_profiles": [],
                }
            else:
                existing = grouped[case_key]
                if existing["short_profile"] != short_profile.strip():
                    raise ValueError(f"Inconsistent short_profile found for case id={case_key}")
                if existing.get("source_title") != item.get("source_title"):
                    raise ValueError(f"Inconsistent source_title found for case id={case_key}")

            grouped[case_key]["generated_profiles"].append(long_profile.strip())

    merged_rows: List[Dict[str, Any]] = []
    sorted_cases = sorted(grouped.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])

    for idx, (_, payload) in enumerate(sorted_cases, start=1):
        merged_rows.append(
            {
                "short_profile_id": f"case_{idx:03d}",
                "short_profile": payload["short_profile"],
                "generated_profiles": payload["generated_profiles"],
                "original_id": payload["original_id"],
                "source_title": payload["source_title"],
            }
        )

    return merged_rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multiple long-profile JSONL files into one grouped JSONL file."
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help="Glob pattern for input JSONL files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_paths = sorted(glob.glob(args.input_glob))
    if not input_paths:
        raise ValueError(f"No files matched --input-glob={args.input_glob}")

    merged_rows = merge_profiles(input_paths)
    write_jsonl(args.output, merged_rows)
    print(f"Merged {len(input_paths)} files into {len(merged_rows)} grouped rows: {args.output}")


if __name__ == "__main__":
    main()
