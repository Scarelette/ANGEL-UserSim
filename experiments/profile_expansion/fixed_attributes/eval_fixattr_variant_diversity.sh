#!/bin/bash
# Evaluate fixattr variant transcripts on the four diversity metrics, grouped by
# variant_id (each variant_id is its own unit; its N runs are the conversations
# diversity is measured across).
#
# Approach: re-key each record so profile_id := variant_id (original kept as
# base_profile_id), because the diversity scorers group by (model, profile_id).
# Output profiles are then identified by variant_id.
#
# Usage (from anywhere):  bash experiments/profile_expansion/fixed_attributes/eval_fixattr_variant_diversity.sh [RUN_NUMBER]
#   RUN_NUMBER = how many runs per variant to incorporate (default 4).
#   Requires the input jsonl to already contain >= RUN_NUMBER runs per variant.
set -euo pipefail

# Run from the repository root.
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

RUN_NUMBER="${1:-4}"
RES="${ANGEL_OUTPUT_DIR:-outputs}/profile_expansion/fixed_attributes"
METRICS="semantic_diversity,behavior_diversity,group_diversity,min_distance_diversity"

for m in angel eeyore patient_psi roleplay_doh; do
  SRC="${SRC_OVERRIDE:-$RES/fixattr_variant_${m}_runs${RUN_NUMBER}.jsonl}"
  KEYED="$RES/fixattr_variant_${m}.variantkeyed.jsonl"
  OUT="$RES/fixattr_variant_${m}.metrics.run${RUN_NUMBER}.json"

  if [[ ! -f "$SRC" ]]; then
    echo "[skip] missing $SRC"; continue
  fi

  echo "[rekey] $SRC -> $KEYED (profile_id := variant_id)"
  python3 - "$SRC" "$KEYED" <<'PY'
import json, sys
from collections import Counter
src, dst = sys.argv[1], sys.argv[2]
runs_per_variant = Counter()
n = 0
with open(src) as f, open(dst, "w") as g:
    for line in f:
        if not line.strip():
            continue
        r = json.loads(line)
        vid = r.get("variant_id")
        if vid is None:
            raise SystemExit(f"record without variant_id in {src}")
        r["base_profile_id"] = r.get("base_profile_id", r.get("profile_id"))
        r["profile_id"] = vid          # group / identify by variant
        g.write(json.dumps(r) + "\n")
        runs_per_variant[vid] += 1
        n += 1
dist = Counter(runs_per_variant.values())
print(f"  {n} records, {len(runs_per_variant)} variant_ids, runs-per-variant distribution={dict(sorted(dist.items()))}")
PY

  echo "[eval] $KEYED -> $OUT (run-range $RUN_NUMBER $RUN_NUMBER)"
  python3 -m experiments.profile_expansion.evaluate_metrics \
    --input  "$KEYED" \
    --output "$OUT" \
    --metrics "$METRICS" \
    --run-range "$RUN_NUMBER" "$RUN_NUMBER" \
    --workers 8 --resume
done

echo "Done. Metrics written to $RES/fixattr_variant_<model>.metrics.run${RUN_NUMBER}.json"
