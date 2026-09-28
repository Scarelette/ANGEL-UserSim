#!/usr/bin/env python3
"""Merge metric sections from several ``evaluate_metrics`` outputs into one file.

The main results (``report_main_results``) read one JSON per model with every metric section present
(``results/clean/<model>_agenda_runs5.metrics.combined.run<N>.json``). In the
original runs, profile alignment and the four diversity metrics were computed
in separate ``evaluate_metrics`` passes over the same transcripts; this script
takes the sections from each input and writes them into one document. Later
inputs override earlier ones for a section that appears in both.

Example (from the repository root):
    python -m experiments.profile_expansion.combine_metrics \
        --inputs outputs/profile_expansion/results/angel_agenda_runs5.alignment.metrics.run4.json \
                 outputs/profile_expansion/results/angel_agenda_runs5.diversity.metrics.run4.json \
        --output outputs/profile_expansion/results/clean/angel_agenda_runs5.metrics.combined.run4.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SECTIONS = (
    "semantic_diversity",
    "behavior_diversity",
    "min_distance_diversity",
    "group_diversity",
    "profile_alignment",
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    combined = {"num_records": None}
    for path in args.inputs:
        doc = json.loads(path.read_text(encoding="utf-8"))
        n = doc.get("num_records")
        if combined["num_records"] is None:
            combined["num_records"] = n
        elif n is not None and n != combined["num_records"]:
            raise SystemExit(
                f"{path}: num_records={n} differs from {combined['num_records']}; "
                "inputs must cover the same transcripts/run count."
            )
        for key in SECTIONS:
            if key in doc:
                combined[key] = doc[key]

    missing = [k for k in SECTIONS if k not in combined]
    if missing:
        print(f"[warn] sections missing from all inputs: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
