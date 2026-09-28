#!/usr/bin/env python3
"""Merge the existing single fixattr transcript (run 1) with newly generated
agenda runs into one variant-keyed file (profile_id := variant_id).

The diversity scorers group by (model, profile_id) and count records as runs, so
after this merge each variant has 4 records => 4 runs.

Usage:
  python merge_existing_plus_runs.py --existing EXISTING.jsonl --new NEW.jsonl --out OUT.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def rekey(record: dict) -> dict:
    """Set profile_id := variant_id (keep original int as base_profile_id)."""
    vid = record.get("variant_id")
    if not vid:
        # agenda runs already carry the variant id in profile_id
        vid = record.get("profile_id")
    record["base_profile_id"] = record.get("base_profile_id", record.get("profile_id"))
    record["profile_id"] = vid
    record["variant_id"] = vid
    return record


def load(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--existing", type=Path, required=True, help="Existing 1-run fixattr file.")
    ap.add_argument("--new", type=Path, required=True, help="Newly generated agenda runs file.")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    runs = Counter()
    n = 0
    with args.out.open("w") as g:
        for src in (args.existing, args.new):
            if not src.exists():
                print(f"[warn] missing {src}; skipping")
                continue
            for rec in load(src):
                rec = rekey(rec)
                g.write(json.dumps(rec) + "\n")
                runs[rec["profile_id"]] += 1
                n += 1

    dist = Counter(runs.values())
    print(f"wrote {args.out}: {n} records, {len(runs)} variant_ids, "
          f"runs-per-variant distribution={dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
