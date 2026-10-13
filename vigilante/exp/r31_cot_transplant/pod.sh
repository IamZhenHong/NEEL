#!/usr/bin/env bash
# R31 on one A100 80GB: Gemma 4 31B, then Qwen3.6-27B (separate processes; the model loads twice).
#   uv run python infra/launch_run.py --name vig-r31 --max-minutes M --max-cost C \
#       --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir vigilante/exp/r31_cot_transplant/results -- bash vigilante/exp/r31_cot_transplant/pod.sh
# Smoke: LIMIT=2 bash vigilante/exp/r31_cot_transplant/pod.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r31_cot_transplant
R=$E/results
mkdir -p $R
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m vigilante.exp.r31_cot_transplant.gen --model google/gemma-4-31B-it --fmt gemma4 --mml 8192 \
    --out $R/gemma-4-31B-it ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a $R/pod.log || echo "[r31] gemma FAILED" | tee -a $R/pod.log
/root/vllm/bin/python -m vigilante.exp.r31_cot_transplant.gen --model Qwen/Qwen3.6-27B --fmt qwen --mml 16384 \
    --out $R/Qwen3.6-27B ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a $R/pod.log || echo "[r31] qwen FAILED" | tee -a $R/pod.log
echo "[r31] all done" | tee -a $R/pod.log
