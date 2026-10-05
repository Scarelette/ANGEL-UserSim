#!/bin/bash
# Example Slurm job for the real-weights demo. Edit the header for your cluster.
#SBATCH --job-name=angel-demo
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/angel_demo_%j.out
##SBATCH --account=<your-account>
##SBATCH --partition=<gpu-partition>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"          # submit from the repository root
mkdir -p logs
# Model locations (defaults: models/Angel-Observer, models/Angel-Actor)
# export ANGEL_OBSERVER_MODEL=/path/to/Angel-Observer
# export ANGEL_ACTOR_MODEL=/path/to/Angel-Actor
python model_usage/scripts/gpu_demo.py
