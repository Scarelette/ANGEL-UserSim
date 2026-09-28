#!/bin/bash
# Fixed-attribute experiment: extra agenda runs per variant, sharded over GPUs,
# then merged with the first run into a 4-run file per model.
# Replaces the original run_fixattr_agenda_gpu.slurm / run_fixattr_array.slurm.
#
#   MODELS="angel eeyore" sbatch experiments/profile_expansion/scripts/fixattr_runs.sbatch
#
# Prerequisites (see the module README, "Fixed-attribute experiment"):
#   $RES/fixattr_variant_<model>.jsonl          first run  (run_fixattr_variant_experiment)
#   $RES/fixattr_variant_short_profiles.jsonl   variant inputs (fixed_attributes.export_variant_inputs)
#SBATCH --job-name=angel_fixattr
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
mkdir -p logs

MODELS="${MODELS:-angel eeyore patient_psi roleplay_doh}"
NGPU="${NGPU:-4}"
NEW_RUNS="${NEW_RUNS:-3}"                       # 1 existing + 3 new = 4 runs per variant
RES="${ANGEL_OUTPUT_DIR:-outputs}/profile_expansion/fixed_attributes"
IN="$RES/fixattr_variant_short_profiles.jsonl"

N=$(grep -c . "$IN")
SH=$(( (N + NGPU - 1) / NGPU ))
echo "input=$IN N=$N shards=$NGPU per-shard=$SH new_runs=$NEW_RUNS"

for M in $MODELS; do
  echo "=== model=$M ==="
  for (( g=0; g<NGPU; g++ )); do
    CUDA_VISIBLE_DEVICES=$g python3 -m experiments.profile_expansion.run_agenda_experiment \
      --model "$M" \
      --input "$IN" \
      --output "$RES/fixattr_variant_${M}_agenda_runs${NEW_RUNS}.shard${g}.jsonl" \
      --num-runs "$NEW_RUNS" \
      --start $((g * SH)) --limit "$SH" \
      --resume &
  done
  wait
  cat "$RES"/fixattr_variant_${M}_agenda_runs${NEW_RUNS}.shard*.jsonl \
    > "$RES/fixattr_variant_${M}_agenda_runs${NEW_RUNS}.jsonl"

  python3 -m experiments.profile_expansion.fixed_attributes.merge_existing_plus_runs \
    --existing "$RES/fixattr_variant_${M}.jsonl" \
    --new      "$RES/fixattr_variant_${M}_agenda_runs${NEW_RUNS}.jsonl" \
    --out      "$RES/fixattr_variant_${M}_runs4.jsonl"
done
echo "done: $RES/fixattr_variant_<model>_runs4.jsonl"
