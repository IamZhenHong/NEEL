#!/usr/bin/env bash
# Stage 0 pod (one A100 80GB): vLLM, Gemma 3 27B bf16, 16 histories + 512 branches. No training.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=welfare_training/results/stage0
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[wt0] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m welfare_training.stage0 run --out $OUT 2>&1 | tee $OUT/run.log
echo "[wt0] pod done" | tee -a $OUT/run.log
