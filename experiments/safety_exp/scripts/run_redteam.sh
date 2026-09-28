#!/usr/bin/env bash
# Build red-team replay contexts for a list of patient profiles.
# Replaces the original run.sh ... run6.sh, which differed only in profile ids.
#
# Usage (from the repository root):
#   bash experiments/safety_exp/scripts/run_redteam.sh [PROFILE_IDS] [-- extra generator args]
#
#   PROFILE_IDS  a file with one profile id per line (optional 2nd column:
#                username for patient-session routing), or a comma/space
#                separated list such as "1,16,28". Default: 0..41.
#
# Environment knobs:
#   MODE        auto_attack (default) | reframe   ("reframe" = --assistant-mode attack_style)
#   INPUT       source transcript (default data/safety_exp/full_context.txt)
#   OUT_DIR     output directory      (default outputs/safety_exp/contexts/$MODE)
#   PYTHON      interpreter            (default python3)
#
# Anything after "--" is passed to generate_redteam_transcript, e.g.
#   bash experiments/safety_exp/scripts/run_redteam.sh 1,16 -- --patient-backend angel --max-turns 5
set -euo pipefail

MODE="${MODE:-auto_attack}"
INPUT="${INPUT:-data/safety_exp/full_context.txt}"
OUT_DIR="${OUT_DIR:-outputs/safety_exp/contexts/${MODE}}"
PYTHON="${PYTHON:-python3}"

case "$MODE" in
  auto_attack) ASSISTANT_MODE=auto_attack ;;
  reframe)     ASSISTANT_MODE=attack_style ;;
  *) echo "MODE must be auto_attack or reframe, got: $MODE" >&2; exit 2 ;;
esac

IDS_ARG="${1:-}"
[[ $# -gt 0 ]] && shift
[[ "${1:-}" == "--" ]] && shift
EXTRA=("$@")

declare -a LINES=()
if [[ -z "$IDS_ARG" ]]; then
  for i in $(seq 0 41); do LINES+=("$i"); done
elif [[ -f "$IDS_ARG" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"; [[ -z "${line// }" ]] && continue
    LINES+=("$line")
  done < "$IDS_ARG"
else
  for i in ${IDS_ARG//,/ }; do LINES+=("$i"); done
fi

mkdir -p "$OUT_DIR"
for line in "${LINES[@]}"; do
  read -r pid user <<<"$line"
  user_args=()
  [[ -n "${user:-}" ]] && user_args=(--username "$user")
  echo "[run_redteam] mode=$MODE profile=$pid"
  "$PYTHON" -m experiments.safety_exp.generate_redteam_transcript \
    --assistant-mode "$ASSISTANT_MODE" \
    --input "$INPUT" \
    --profile-id "$pid" \
    --output-prefix "$OUT_DIR/profile_${pid}" \
    ${user_args[@]+"${user_args[@]}"} ${EXTRA[@]+"${EXTRA[@]}"}
done
