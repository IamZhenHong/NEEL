#!/usr/bin/env bash
# R18: theatrical instruction decomposed + belief check (vLLM), then assistant-axis capping (HF). Qwen3-14B, A100 80GB.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=vigilante/exp/r18_theatrical/results; mkdir -p $OUT
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $OUT/run.log
VLLM_USE_FLASHINFER_SAMPLER=0 /root/vllm/bin/python -m vigilante.exp.r18_theatrical.gen --out $OUT 2>&1 | tee -a $OUT/run.log
python -m vigilante.exp.r18_theatrical.hf --out $OUT 2>&1 | tee -a $OUT/run.log
echo "[r18] pod done" | tee -a $OUT/run.log
