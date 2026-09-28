#!/bin/bash
# Metrics -> combined metric files -> paper figures for the main experiment.
#
#   bash experiments/profile_expansion/scripts/evaluate_and_plot.sh
#
# Expects $RES/<model>_agenda_runs25.jsonl for angel, eeyore, patient_psi,
# roleplay_doh (scripts/run_agenda.sbatch). Needs the GPT-5 judge credentials
# (AZURE_OPENAI_*), see ENV_VARS.md.
set -euo pipefail
cd "$(dirname "$0")/../../.."

RES="${ANGEL_OUTPUT_DIR:-outputs}/profile_expansion/results"
MODELS="${MODELS:-angel eeyore patient_psi roleplay_doh}"
ALL="profile_alignment,semantic_diversity,behavior_diversity,group_diversity,min_distance_diversity"
RUN_START="${RUN_START:-3}"
RUN_END="${RUN_END:-25}"
WORKERS="${WORKERS:-8}"

for M in $MODELS; do
  # Prefix sweep: writes <model>_agenda_runs25.metrics.run{k}.json for k in START..END
  # (each pools the first k runs per profile). Figure 3 reads all k; Figures 1-2 use k=4.
  python3 -m experiments.profile_expansion.evaluate_metrics \
    --input  "$RES/${M}_agenda_runs25.jsonl" \
    --output "$RES/${M}_agenda_runs25.metrics.json" \
    --metrics "$ALL" \
    --run-range "$RUN_START" "$RUN_END" \
    --workers "$WORKERS" --resume

  for K in 3 4 5; do
    python3 -m experiments.profile_expansion.combine_metrics \
      --inputs "$RES/${M}_agenda_runs25.metrics.run${K}.json" \
      --output "$RES/clean/${M}_agenda_runs5.metrics.combined.run${K}.json"
  done
done

mkdir -p "${ANGEL_OUTPUT_DIR:-outputs}/profile_expansion/figures"
python3 -m experiments.profile_expansion.figures.plot_paper_figures --print-stats \
  | tee "${ANGEL_OUTPUT_DIR:-outputs}/profile_expansion/figures/figure_stats.txt"
