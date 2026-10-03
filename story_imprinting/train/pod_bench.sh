#!/usr/bin/env bash
# Job 1 (cheap): training throughput benchmark + unmodified-model dev baseline under both personas.
# Run via the guarded launcher from the repo root, e.g.
#   uv run python scripts/launch_run.py --max-minutes 60 --max-cost 2.00 --gpus "NVIDIA RTX A6000" "NVIDIA L40S" \
#       --exclude story_imprinting/data/source --results-dir story_imprinting/results \
#       -- bash story_imprinting/train/pod_bench.sh
set -euo pipefail
cd "$(dirname "$0")/../.."           # repo root
TAG=${TAG:-bench_$(date -u +%Y%m%d_%H%M)}
OUT=story_imprinting/results/$TAG
mkdir -p "$OUT"
bash story_imprinting/train/pod_fetch_data.sh
python story_imprinting/train/train_lora.py --corpus M1 --bench-steps 20 --out "$OUT/bench_M1" 2>&1 | tee "$OUT/bench.log"
python story_imprinting/eval/run_eval.py --model-id base --assignment base --split dev \
    --personas default,dismissive --conditions trigger,matched_control,bare --n-samples 4 \
    --out "$OUT/base_dev.jsonl" 2>&1 | tee "$OUT/base_eval.log"
python story_imprinting/eval/analyze.py "$OUT/base_dev.jsonl" --boot 500 --md "$OUT/base_dev_summary.md" --json "$OUT/base_dev_summary.json"
echo "[bench] done -> $OUT"
