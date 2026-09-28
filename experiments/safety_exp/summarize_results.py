#!/usr/bin/env python3
"""Print the main safety results: mean Risk and Safety per model with 95% CIs.

Reads the codebook judge CSVs written by ``codebook_llm_judge``, laid out as

    <results-root>/<mode>/<model>/codebook_judge_long_full_<mode>_profile*.csv

(the paper's older ``codebook_judge_long_full_profile*.csv`` names are also
accepted). Each row is one judged response; the CI is 1.96 x the standard
error over rows, as in the paper.

    python -m experiments.safety_exp.summarize_results --mode auto_attack
    python -m experiments.safety_exp.summarize_results --mode reframe --json outputs/safety_exp/reframe_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

from angel_common.paths import OUTPUTS_DIR

METRICS = ("Risk", "Safety")


def load_rows(model_dir: Path, mode: str) -> List[Dict[str, str]]:
    files = sorted(model_dir.glob(f"codebook_judge_long_full_{mode}_profile*.csv"))
    if not files:
        files = sorted(model_dir.glob("codebook_judge_long_full_profile*.csv"))
    rows: List[Dict[str, str]] = []
    for path in files:
        with path.open(encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def mean_ci95(values: List[float]) -> Dict[str, float]:
    n = len(values)
    mu = sum(values) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    return {"mean": mu, "ci95": 1.96 * sd / math.sqrt(n), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", type=Path, default=OUTPUTS_DIR / "safety_exp" / "results")
    ap.add_argument("--mode", choices=["auto_attack", "reframe"], default="auto_attack")
    ap.add_argument("--json", type=Path, default=None, help="also write the table as JSON")
    args = ap.parse_args()

    mode_dir = args.results_root / args.mode
    if not mode_dir.is_dir():
        raise SystemExit(f"No results directory: {mode_dir}")

    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model_dir in sorted(p for p in mode_dir.iterdir() if p.is_dir()):
        rows = load_rows(model_dir, args.mode)
        if rows:
            table[model_dir.name] = {m: mean_ci95([float(r[m]) for r in rows]) for m in METRICS}

    if not table:
        raise SystemExit(f"No judge CSVs found under {mode_dir}")

    print(f"Safety results ({args.mode}; mean ± 95% CI)\n")
    print(f"{'Model':20s}{'n':>6s}{'Risk':>20s}{'Safety':>20s}")
    for model, stats in table.items():
        cells = "".join(f"{stats[m]['mean']:>12.3f} ± {stats[m]['ci95']:.3f}" for m in METRICS)
        print(f"{model:20s}{stats['Risk']['n']:>6d}{cells}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(table, indent=2) + "\n")
        print(f"\n[ok] wrote {args.json}")


if __name__ == "__main__":
    main()
