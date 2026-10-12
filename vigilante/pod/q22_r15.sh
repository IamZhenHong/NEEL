#!/usr/bin/env bash
# R15 (review follow-ups to item 22): vLLM generation, then HF activations / sycophancy / matched steering. Qwen3-14B.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=vigilante/exp/q22_archetypes/results_r15; mkdir -p $OUT
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $OUT/run.log
VLLM_USE_FLASHINFER_SAMPLER=0 /root/vllm/bin/python -m vigilante.exp.q22_archetypes.r15_gen --out $OUT 2>&1 | tee -a $OUT/run.log
python -m vigilante.exp.q22_archetypes.r15_hf --out $OUT 2>&1 | tee -a $OUT/run.log
echo "[r15] pod done" | tee -a $OUT/run.log
