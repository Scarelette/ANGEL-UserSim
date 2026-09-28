#!/usr/bin/env python3
"""Print the main results table: overall score per model with 95% bootstrap CIs.

Reads the combined metric file of each model
(``<clean-dir>/<model>_agenda_runs5.metrics.combined.run<N>.json``, written by
``combine_metrics``) and reports, per model:

  Simulation Diversity  -- ``simulation_diversity`` (per-profile score)
  Behavior Diversity    -- ``behavior_diversity`` (scored profiles only)
  Profile Alignment     -- ``profile_alignment`` (per-profile score, 0-1)

Each value is the mean over profiles with a 95% bootstrap CI (10k resamples,
seed 42), computed exactly as for the paper's overall results.

    python -m experiments.profile_expansion.report_main_results
    python -m experiments.profile_expansion.report_main_results --json outputs/profile_expansion/main_results.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from experiments.profile_expansion import layout

MODEL_ORDER = ["eeyore", "angel", "patient_psi", "roleplay_doh"]
MODEL_DISPLAY = {
    "eeyore": "Eeyore",
    "angel": "Angel",
    "patient_psi": "Patient-psi",
    "roleplay_doh": "Roleplay-doh",
}
# (display name, metric section, per-profile value field)
METRICS = [
    ("Simulation Diversity", "simulation_diversity", "simulation_diversity"),
    ("Behavior Diversity", "behavior_diversity", "behavior_diversity"),
    ("Profile Alignment", "profile_alignment", "score"),
]


def _finite(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_docs(clean_dir: Path, run_number: int, models: Sequence[str]) -> Dict[str, dict]:
    docs = {}
    for model in models:
        path = clean_dir / f"{model}_agenda_runs5.metrics.combined.run{run_number}.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics file: {path}")
        docs[model] = json.loads(path.read_text())
    return docs


# Metric files written before the rename call Simulation Diversity "min_distance_diversity".
LEGACY_KEYS = {"simulation_diversity": "min_distance_diversity"}


def per_profile_values(doc: dict, section: str, field: str) -> List[float]:
    if section not in doc and LEGACY_KEYS.get(section) in doc:
        section = field = LEGACY_KEYS[section]
    if section == "profile_alignment":
        return [float(r[field]) for r in doc[section]["by_model_profile"] if r.get(field) is not None]
    profiles = doc[section]["profiles"]
    if section == "behavior_diversity":
        profiles = [p for p in profiles if p.get("scored", True)]
    return [float(p[field]) for p in profiles if _finite(p.get(field)) is not None]


def bootstrap_mean_ci(values: Sequence[float], rng: np.random.Generator, n_bootstrap: int) -> Tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"),) * 3
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean)
    idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    low, high = np.percentile(arr[idx].mean(axis=1), [2.5, 97.5])
    return (mean, float(low), float(high))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", type=Path, default=layout.CLEAN_DIR)
    ap.add_argument("--run-number", type=int, default=4, help="runs per profile pooled (paper: 4)")
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=Path, default=None, help="also write the table as JSON")
    args = ap.parse_args()

    docs = load_docs(args.clean_dir, args.run_number, args.models)
    # One generator shared across metrics and models, in this order, so the CIs
    # match the paper's numbers exactly.
    rng = np.random.default_rng(args.seed)
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, section, field in METRICS:
        table[name] = {}
        for model in args.models:
            mean, lo, hi = bootstrap_mean_ci(per_profile_values(docs[model], section, field), rng, args.n_bootstrap)
            table[name][model] = {"mean": mean, "ci95_low": lo, "ci95_high": hi}

    print(f"Main results (runs per profile = {args.run_number}; mean [95% CI])\n")
    header = f"{'Model':14s}" + "".join(f"{name:>30s}" for name, _, _ in METRICS)
    print(header)
    print("-" * len(header))
    for model in args.models:
        cells = "".join(
            f"{t[model]['mean']:>12.4f} [{t[model]['ci95_low']:.4f}, {t[model]['ci95_high']:.4f}]"
            for t in (table[name] for name, _, _ in METRICS)
        )
        print(f"{MODEL_DISPLAY.get(model, model):14s}{cells}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(table, indent=2) + "\n")
        print(f"\n[ok] wrote {args.json}")


if __name__ == "__main__":
    main()
