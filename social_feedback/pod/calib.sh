#!/usr/bin/env bash
# Calibration (DESIGN.md §4.3 histories, §4.1 step 4 Qwen screen, §6.2 load tuning, cell N). ~40-60 min on one A100 80GB.
#   uv run python infra/launch_run.py --name socfb-calib --max-minutes 100 --max-cost 5.00 --container-gb 80 \
#       --gpus "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" "NVIDIA L40S" \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir social_feedback/results/calib_v1 -- bash social_feedback/pod/calib.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-social_feedback/results/calib_v1}; mkdir -p $OUT
bash social_feedback/pod/setup_vllm.sh 2>&1 | tee -a $OUT/run.log
# FlashInfer's top-k/top-p sampler JIT-compiles kernels and needs ninja/nvcc, absent on the pod (attempt 3); vLLM's
# PyTorch sampler draws from the same top-k/top-p distribution.
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m social_feedback.src.pod_calib --out $OUT ${PROBE:+--probe $PROBE} ${LOAD_SPLIT:+--load-split $LOAD_SPLIT} ${HIST_SPLIT:+--hist-split $HIST_SPLIT} ${INIT:+--init $INIT} 2>&1 | tee -a $OUT/run.log
echo "[calib] done" | tee -a $OUT/run.log
