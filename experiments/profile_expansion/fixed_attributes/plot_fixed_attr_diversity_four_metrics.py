#!/usr/bin/env python3
"""Fixed-attribute-number vs diversity, for all four diversity metrics.

Uses the fixed-attribute variant experiment results:
  $ANGEL_OUTPUT_DIR/profile_expansion/fixed_attributes/fixattr_variant_<model>.metrics.run4.json

Each variant profile_id encodes  <base_profile>_k<K>_s<S>  where K is the number
of FIXED (revealed) attributes. For each (model, K) we average the profile-level
diversity score across the base-profile variants and draw a bootstrap 95% CI band.

Hypothesis: the more attributes are fixed, the more constrained the patient, so a
valid diversity metric should DECREASE as K grows.

Produces one figure per metric (semantic / group / simulation / behavior) in
--output-dir, plotting the measured scores.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from collections import defaultdict
from pathlib import Path
from experiments.profile_expansion import layout
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from experiments.profile_expansion.figures import acl_fig_style as st  # noqa: E402

MODEL_ORDER = ["eeyore", "angel", "patient_psi", "roleplay_doh"]
MODEL_DISPLAY = {
    "eeyore": "Eeyore", "angel": "Angel",
    "patient_psi": "Patient-psi", "roleplay_doh": "Roleplay-doh",
}
MODEL_COLOR = {
    "eeyore": "#ffbe78", "angel": "#1f77b4",
    "patient_psi": "#98df8a", "roleplay_doh": "#f28e8c",
}

# (section, profile-level score field, skip-unscored, display title)
METRICS = [
    ("semantic_diversity", "semantic_diversity", False, "Semantic Diversity"),
    ("group_diversity", "group_diversity", False, "Group Diversity"),
    ("min_distance_diversity", "min_distance_diversity", False, "Simulation Diversity"),
    ("behavior_diversity", "behavior_diversity", True, "Behavior Diversity"),
]
OUTPREFIX = {
    "semantic_diversity": "fixed_attr_vs_semantic_diversity_run4",
    "group_diversity": "fixed_attr_vs_group_diversity_run4",
    "min_distance_diversity": "fixed_attr_vs_simulation_diversity_run4",
    "behavior_diversity": "fixed_attr_vs_behavior_diversity_run4",
}

# Y-axis label per metric (no in-figure titles any more).
YLABEL = {
    "semantic_diversity": "Semantic diversity",
    "group_diversity": "Group diversity",
    "min_distance_diversity": "Simulation diversity",
    "behavior_diversity": "Behavior diversity",
}

# Printed size in inches, per target width.
FIG_HEIGHT = {"column": 2.2, "text": 2.6}

VARIANT_RE = re.compile(r"^(.+)_k(\d+)_s(\d+)$")


def parse_model(path: Path) -> str:
    m = re.match(r"fixattr_variant_(.+)\.metrics\.run\d+\.json$", path.name)
    return m.group(1) if m else path.stem


def collect(paths: List[Path], section: str, field: str, skip_unscored: bool):
    """(model, k) -> list of profile scores across base-profile variants."""
    grouped: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for path in paths:
        model = parse_model(path)
        doc = json.loads(path.read_text(encoding="utf-8"))
        for prof in doc.get(section, {}).get("profiles", []) or []:
            if skip_unscored and not prof.get("scored", True):
                continue
            m = VARIANT_RE.match(str(prof.get("profile_id", "")))
            if not m:
                continue
            k = int(m.group(2))
            val = prof.get(field)
            if val is None:
                continue
            try:
                grouped[(model, k)].append(float(val))
            except (TypeError, ValueError):
                continue
    return grouped


def boot_ci(values: List[float], n_boot: int, rng: np.random.Generator) -> Tuple[float, float, float, int]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    mean = float(arr.mean())
    if arr.size == 1:
        return (mean, mean, mean, 1)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (mean, float(lo), float(hi), int(arr.size))


def plot_metric(section, title, grouped, outdir, x_min, x_max, n_boot, seed):
    rng = np.random.default_rng(seed)
    models = sorted({m for (m, _k) in grouped}, key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    ks = sorted({k for (_m, k) in grouped if x_min <= k <= x_max})

    fig, ax = plt.subplots(figsize=(11, 7))
    csv_rows = []
    for model in models:
        xs, ys, los, his = [], [], [], []
        for k in ks:
            vals = grouped.get((model, k), [])
            if not vals:
                continue
            mean, lo, hi, n = boot_ci(vals, n_boot, rng)
            xs.append(k); ys.append(mean); los.append(lo); his.append(hi)
            csv_rows.append({"metric": section, "model": model, "fixed_attribute_number": k,
                             "num_profiles": n, "mean": mean, "ci_low": lo, "ci_high": hi})
        if not xs:
            continue
        color = MODEL_COLOR.get(model, None)
        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=6.5, color=color,
                label=MODEL_DISPLAY.get(model, model), zorder=3)
        ax.fill_between(xs, los, his, color=color, alpha=0.18, zorder=2, linewidth=0)

    ax.set_xticks(ks)
    ax.set_xlim(min(ks) - 0.5, max(ks) + 0.5)
    ax.set_xlabel("Fixed Attribute Number", fontsize=13)
    ax.set_ylabel(f"Average {title} Score", fontsize=13)
    ax.set_title(f"Fixed Attribute Number vs. {title}", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, title="Model")
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    prefix = OUTPREFIX[section]
    png = outdir / f"{prefix}.png"
    fig.savefig(png, dpi=300)
    fig.savefig(outdir / f"{prefix}.pdf")
    plt.close(fig)

    with (outdir / f"{prefix}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "model", "fixed_attribute_number",
                                           "num_profiles", "mean", "ci_low", "ci_high"])
        w.writeheader()
        for r in sorted(csv_rows, key=lambda r: (MODEL_ORDER.index(r["model"]) if r["model"] in MODEL_ORDER else 99,
                                                  r["fixed_attribute_number"])):
            w.writerow(r)
    print(f"[{section}] wrote {png}", flush=True)


def make_bins(ks: List[int], num_bins: int) -> List[Tuple[int, int, str]]:
    """Split the sorted distinct k values into num_bins contiguous ranges."""
    uniq = sorted(set(ks))
    chunks = np.array_split(np.asarray(uniq), num_bins)
    bins = []
    for ch in chunks:
        if len(ch) == 0:
            continue
        lo, hi = int(ch[0]), int(ch[-1])
        bins.append((lo, hi, f"{lo}–{hi}" if lo != hi else f"{lo}"))
    return bins


def plot_metric_binned(section, title, grouped, outdir, x_min, x_max, num_bins, n_boot, seed):
    rng = np.random.default_rng(seed)
    models = sorted({m for (m, _k) in grouped},
                    key=lambda m: MODEL_ORDER.index(m) if m in MODEL_ORDER else 99)
    ks = [k for (_m, k) in grouped if x_min <= k <= x_max]
    bins = make_bins(ks, num_bins)
    centers = np.arange(len(bins), dtype=float)

    fig, ax = plt.subplots(figsize=(11, 7))
    csv_rows = []
    for model in models:
        xs, ys, lo_err, hi_err = [], [], [], []
        for bi, (lo, hi, _lbl) in enumerate(bins):
            vals: List[float] = []
            for k in range(lo, hi + 1):
                vals.extend(grouped.get((model, k), []))
            if not vals:
                continue
            mean, ci_lo, ci_hi, n = boot_ci(vals, n_boot, rng)
            xs.append(bi); ys.append(mean)
            lo_err.append(mean - ci_lo); hi_err.append(ci_hi - mean)
            csv_rows.append({"metric": section, "model": model, "bin": f"{lo}-{hi}",
                             "num_profiles": n, "mean": mean, "ci_low": ci_lo, "ci_high": ci_hi})
        if not xs:
            continue
        color = MODEL_COLOR.get(model, None)
        ax.errorbar(xs, ys, yerr=[lo_err, hi_err], marker="o", linewidth=2.4, markersize=8,
                    capsize=5, elinewidth=1.8, color=color,
                    label=MODEL_DISPLAY.get(model, model), zorder=3)

    ax.set_xticks(centers)
    ax.set_xticklabels([b[2] for b in bins])
    ax.set_xlim(-0.4, len(bins) - 0.6)
    ax.set_xlabel("Fixed Attribute Number (binned)", fontsize=13)
    ax.set_ylabel(f"Average {title} Score", fontsize=13)
    ax.set_title(f"Fixed Attribute Number vs. {title} (binned)", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, title="Model")
    fig.tight_layout()

    outdir.mkdir(parents=True, exist_ok=True)
    prefix = OUTPREFIX[section] + "_binned"
    fig.savefig(outdir / f"{prefix}.png", dpi=300)
    fig.savefig(outdir / f"{prefix}.pdf")
    plt.close(fig)
    with (outdir / f"{prefix}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["metric", "model", "bin", "num_profiles",
                                           "mean", "ci_low", "ci_high"])
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)
    print(f"[{section}] wrote {outdir / (prefix + '.png')}", flush=True)


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-glob", default=str(layout.FIXATTR_DIR / "fixattr_variant_*.metrics.run4.json"))
    ap.add_argument("--output-dir", type=Path, default=layout.FIXATTR_DIR / "fig")
    ap.add_argument("--x-min", type=int, default=1)
    ap.add_argument("--x-max", type=int, default=22)
    ap.add_argument("--n-bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--binned", action="store_true", help="Bin k into ranges instead of per-k lines.")
    ap.add_argument("--num-bins", type=int, default=3)
    ap.add_argument("--width", choices=["column", "text"], default="column",
                    help="Printed width: one ACL column, or the two-column span.")
    ap.add_argument("--metrics", nargs="+", default=None,
                    choices=[m[0] for m in METRICS],
                    help="Only draw these metric sections (default: all four).")
    args = ap.parse_args()
    paths = [Path(p) for p in sorted(glob.glob(args.results_glob))]
    if not paths:
        raise FileNotFoundError(args.results_glob)
    print(f"[load] {len(paths)} files: {[parse_model(p) for p in paths]}", flush=True)

    for section, field, skip_unscored, title in METRICS:
        if args.metrics and section not in args.metrics:
            continue
        grouped = collect(paths, section, field, skip_unscored)
        if args.binned:
            plot_metric_binned(section, title, grouped, args.output_dir,
                               args.x_min, args.x_max, args.num_bins, args.n_bootstrap, args.seed)
        else:
            plot_metric(section, title, grouped, args.output_dir,
                        args.x_min, args.x_max, args.n_bootstrap, args.seed)


if __name__ == "__main__":
    main()
