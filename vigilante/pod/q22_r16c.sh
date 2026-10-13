#!/usr/bin/env bash
# R16 continuation of replies that hit their 300-token limit (honesty block, free-text MCQ without the letter-only line).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=vigilante/exp/q22_archetypes/results_r16gpu
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $OUT/run_cont.log
VLLM_USE_FLASHINFER_SAMPLER=0 /root/vllm/bin/python -m vigilante.exp.q22_archetypes.r16_cont --out $OUT 2>&1 | tee -a $OUT/run_cont.log
echo "[r16c] pod done" | tee -a $OUT/run_cont.log
