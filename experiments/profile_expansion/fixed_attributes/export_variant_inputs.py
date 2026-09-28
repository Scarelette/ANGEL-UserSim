#!/usr/bin/env python3
"""Build the agenda-experiment input for the extra fixed-attribute runs.

After ``run_fixattr_variant_experiment`` has produced one transcript per
(profile, fixed_attribute_number) for each model (``fixattr_variant_<model>.jsonl``),
the remaining runs are generated with the ordinary agenda runner over the
*union* of the variants those first runs used. This script writes that union as
``fixattr_variant_short_profiles.jsonl``, one row per variant, sorted by
``variant_id``::

    {"id", "variant_id", "fixed_attribute_number", "source_title", "short_patient_profile"}

This step was done ad hoc in the original research code; the output here was
checked to reproduce the rows of the paper's input file (224 variants).

Usage (from the repository root):
    python -m experiments.profile_expansion.fixed_attributes.export_variant_inputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from experiments.profile_expansion import layout

MODELS = ("angel", "eeyore", "patient_psi", "roleplay_doh")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[layout.FIXATTR_DIR / f"fixattr_variant_{m}.jsonl" for m in MODELS],
        help="First-run fixed-attribute transcripts (one file per model).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=layout.FIXATTR_DIR / "fixattr_variant_short_profiles.jsonl",
    )
    args = ap.parse_args()

    variants = {}
    for path in args.inputs:
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                vid = row.get("variant_id")
                if vid and vid not in variants:
                    variants[vid] = {
                        "id": vid,
                        "variant_id": vid,
                        "fixed_attribute_number": row.get("fixed_attribute_number"),
                        "source_title": row.get("source_title"),
                        "short_patient_profile": row.get("short_patient_profile"),
                    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as g:
        for vid in sorted(variants):
            g.write(json.dumps(variants[vid], ensure_ascii=False) + "\n")
    print(f"wrote {len(variants)} variants -> {args.output}")


if __name__ == "__main__":
    main()
