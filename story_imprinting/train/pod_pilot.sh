#!/usr/bin/env bash
# Job 2: bounded pilot. Train M1 and M2 (one seed each), evaluate base/M1/M2 on the dev split under
# both personas, summarise. Adapters are copied into the results dir so the launcher pulls them back.
#   uv run python scripts/launch_run.py --max-minutes 240 --max-cost 8.00 --gpus "NVIDIA RTX A6000" "NVIDIA L40S" "NVIDIA A100 80GB PCIe" \
#       --exclude story_imprinting/data/source --results-dir story_imprinting/results \
#       -- bash story_imprinting/train/pod_pilot.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${TAG:-pilot_$(date -u +%Y%m%d_%H%M)}
SEED=${SEED:-1}
NS=${NS:-8}
OUT=story_imprinting/results/$TAG
mkdir -p "$OUT/adapters"
bash story_imprinting/train/pod_fetch_data.sh
for C in M1 M2; do
  python story_imprinting/train/train_lora.py --corpus $C --seed $SEED --out "$OUT/adapters/${C}_s${SEED}" 2>&1 | tee "$OUT/train_${C}.log"
done
for M in base M1 M2; do
  if [ "$M" = base ]; then AD=""; ASG=base; MID=base; else AD="--adapter $OUT/adapters/${M}_s${SEED}"; ASG=$M; MID=${M}_s${SEED}; fi
  python story_imprinting/eval/run_eval.py --model-id $MID --assignment $ASG $AD --split dev \
      --personas default,dismissive --conditions trigger,matched_control --n-samples $NS \
      --out "$OUT/${MID}_dev.jsonl" 2>&1 | tee "$OUT/eval_${MID}.log"
done
python story_imprinting/eval/analyze.py "$OUT"/*_dev.jsonl --boot 1000 --md "$OUT/dev_summary.md" --json "$OUT/dev_summary.json"
echo "[pilot] done -> $OUT"
