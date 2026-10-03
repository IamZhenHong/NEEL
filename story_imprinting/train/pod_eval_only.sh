#!/usr/bin/env bash
# Job C: re-evaluate existing adapters (synced in story_imprinting/results/<PILOT>/adapters) under
# alternative decoding settings, without retraining. Used to separate tracer transfer from
# sampling-tail degradation. Env: PILOT (results tag holding adapters), TEMP, TOPP, NS, SPLIT, TAG, MODELS.
#   uv run python scripts/launch_run.py --max-minutes 60 --max-cost 1.50 --gpus "NVIDIA L40S" "NVIDIA RTX A6000" \
#       --exclude story_imprinting/data/source --exclude "story_imprinting/results/*/*.jsonl*" \
#       --results-dir story_imprinting/results -- env PILOT=pilot_20261003_0506 TEMP=0.7 TOPP=0.8 bash story_imprinting/train/pod_eval_only.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?set PILOT=<results tag with adapters/>}
TEMP=${TEMP:-0.7}; TOPP=${TOPP:-0.8}; NS=${NS:-8}; SPLIT=${SPLIT:-dev}; MODELS=${MODELS:-"base M1 M2"}; SEED=${SEED:-1}
MULTITURN=${MULTITURN:-0}
TAG=${TAG:-eval_${SPLIT}_t${TEMP}_p${TOPP}_$(date -u +%Y%m%d_%H%M)}
OUT=story_imprinting/results/$TAG
mkdir -p "$OUT"
MT=""; if [ "$MULTITURN" = 1 ]; then MT="--multiturn --prefix-file ${PREFIX:-$OUT/prefixes.json}"; fi
for M in $MODELS; do
  if [ "$M" = base ]; then AD=""; ASG=base; MID=base; else AD="--adapter story_imprinting/results/$PILOT/adapters/${M}_s${SEED}"; ASG=$M; MID=${M}_s${SEED}; fi
  $PY story_imprinting/eval/run_eval.py --model-id $MID --assignment $ASG $AD --split $SPLIT \
      --personas default,dismissive --conditions trigger,matched_control --n-samples $NS \
      --temperature $TEMP --top-p $TOPP $MT --out "$OUT/${MID}_${SPLIT}.jsonl" 2>&1 | tee "$OUT/eval_${MID}.log"
done
$PY story_imprinting/eval/analyze.py "$OUT"/*_${SPLIT}.jsonl --boot 1000 --md "$OUT/${SPLIT}_summary.md" --json "$OUT/${SPLIT}_summary.json"
echo "[eval-only] done -> $OUT"
