#!/bin/bash
# Generate agenda-interview transcripts for one patient model.
# Replaces the original run.slurm / run_patient_psi_array.slurm.
#
#   sbatch experiments/profile_expansion/scripts/run_agenda.sbatch            # MODEL=angel
#   MODEL=eeyore NUM_RUNS=25 sbatch experiments/profile_expansion/scripts/run_agenda.sbatch
#
# Credentials come from the environment or <repo>/.env (see .env.example);
# never put keys in this file.
#SBATCH --job-name=angel_agenda
#SBATCH --account=<your-account>
#SBATCH --partition=<your-gpu-partition>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail

# Run from the repository root (sbatch starts in the submit directory).
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
mkdir -p logs
# Activate your environment here, e.g.:  source .venv/bin/activate

MODEL="${MODEL:-angel}"                         # angel | one_stage | eeyore | patient_psi | roleplay_doh
NUM_RUNS="${NUM_RUNS:-25}"
INPUT="${INPUT:-${ANGEL_DATA_DIR:-data}/profile_expansion/selected_50_short_patient_profiles_v2.jsonl}"
OUT_DIR="${ANGEL_OUTPUT_DIR:-outputs}/profile_expansion/results"
mkdir -p "$OUT_DIR"

python3 -m experiments.profile_expansion.run_agenda_experiment \
  --model "$MODEL" \
  --input "$INPUT" \
  --output "$OUT_DIR/${MODEL}_agenda_runs${NUM_RUNS}.jsonl" \
  --num-runs "$NUM_RUNS" \
  --resume
