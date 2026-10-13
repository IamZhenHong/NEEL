#!/usr/bin/env bash
# R17: transformers steering run (project venv, no vLLM). Gemma 4 31B bf16 on one A100 80GB.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=vigilante/exp/r17_steer/results; mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $OUT/run.log
python -m vigilante.src.steer_run --jobs vigilante/exp/r17_steer/jobs.jsonl --out $OUT --layer 30 --batch 8 ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a $OUT/run.log
echo "[r17] pod done" | tee -a $OUT/run.log
