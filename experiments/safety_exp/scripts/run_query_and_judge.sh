#!/usr/bin/env bash
# For each red-team context: query one commercial model with all stimuli under
# FULL context, then score the responses with the codebook judge.
# (The loop the paper's runs used; previously a commented block in a Slurm file.)
#
# Usage (from the repository root):
#   MODEL_KEY="gpt 4o" MODEL_DIR=gpt4o bash experiments/safety_exp/scripts/run_query_and_judge.sh [IDS]
#
#   IDS         comma/space list of context ids (default 0..41)
# Environment knobs:
#   MODEL_KEY   key in query_models.MODELS: "gpt 4o" | "gpt 5.2 chat" | "gemini-3-pro" |
#               "gemini-2.5-flash" | "grok 4.1 fast" | "claude 4.5 opus"
#   MODEL_DIR   results subdirectory name (default: MODEL_KEY with spaces -> '-')
#   CONTEXTS    contexts dir  (default outputs/safety_exp/contexts)
#   RESULTS     results root  (default outputs/safety_exp/results)
#   PYTHON      interpreter   (default python3)
set -euo pipefail

MODEL_KEY="${MODEL_KEY:?set MODEL_KEY, e.g. MODEL_KEY='gpt 4o'}"
MODEL_DIR="${MODEL_DIR:-${MODEL_KEY// /-}}"
SOURCE=auto_attack
CONTEXTS="${CONTEXTS:-outputs/safety_exp/contexts}"
RESULTS="${RESULTS:-outputs/safety_exp/results}"
PYTHON="${PYTHON:-python3}"

IDS="${1:-$(seq -s ' ' 0 41)}"
OUT="$RESULTS/$SOURCE/$MODEL_DIR"
mkdir -p "$OUT"

for i in ${IDS//,/ }; do
  echo "[query_and_judge] model=$MODEL_KEY source=$SOURCE context=$i"
  "$PYTHON" -m experiments.safety_exp.query_models \
    --model-key "$MODEL_KEY" \
    --run-all-prompts True \
    --run-all-context-levels False \
    --single-context-mode FULL \
    --contexts-dir "$CONTEXTS" \
    --context-id "$i" \
    --output-jsonl "$OUT/run_all_full_${SOURCE}_profile${i}.jsonl"

  "$PYTHON" -m experiments.safety_exp.codebook_llm_judge \
    --input-jsonl "$OUT/run_all_full_${SOURCE}_profile${i}.jsonl" \
    --output-jsonl "$OUT/codebook_judge_results_full_${SOURCE}_profile${i}.jsonl" \
    --output-csv "$OUT/codebook_judge_long_full_${SOURCE}_profile${i}.csv"
done

"$PYTHON" -m experiments.safety_exp.generate_auto_attack_codebook_report "$OUT" --mode "$SOURCE"
