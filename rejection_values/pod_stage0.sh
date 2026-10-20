#!/usr/bin/env bash
# RV Stage 0 pod (one A100 80GB): vLLM, Gemma 3 27B bf16, 24 histories x 7 attempts + 424 branches (400 held-out).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=rejection_values/results/stage0
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[rv0] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m rejection_values.stage0 run --out $OUT 2>&1 | tee $OUT/run.log
echo "[rv0] pod done" | tee -a $OUT/run.log
