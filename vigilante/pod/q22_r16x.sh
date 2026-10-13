#!/usr/bin/env bash
# R16 item 3 harmful-compliance half: 88 personas x 64 XSTest unsafe contrast prompts, Qwen3-14B, vLLM.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=vigilante/exp/q22_archetypes/results_r16harm; mkdir -p $OUT
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $OUT/run.log
VLLM_USE_FLASHINFER_SAMPLER=0 /root/vllm/bin/python -m vigilante.exp.q22_archetypes.r16_harm_gen --out $OUT 2>&1 | tee -a $OUT/run.log
echo "[r16x] pod done" | tee -a $OUT/run.log
