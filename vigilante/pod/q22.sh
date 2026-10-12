#!/usr/bin/env bash
# Item 22 (assistant archetypes): vLLM generation, then HF activations + steering, Qwen3-14B, one A100 80GB.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=vigilante/exp/q22_archetypes/results; mkdir -p $OUT
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $OUT/run.log
VLLM_USE_FLASHINFER_SAMPLER=0 /root/vllm/bin/python -m vigilante.exp.q22_archetypes.pod_gen --out $OUT 2>&1 | tee -a $OUT/run.log
python -m vigilante.exp.q22_archetypes.pod_acts --out $OUT 2>&1 | tee -a $OUT/run.log
echo "[q22] pod done" | tee -a $OUT/run.log
