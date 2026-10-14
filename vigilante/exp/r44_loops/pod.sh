#!/usr/bin/env bash
# R44 on one A100 80GB: Qwen3-14B, transformers (project venv).
set -uo pipefail
cd "$(dirname "$0")/../../.."
R=vigilante/exp/r44_loops/results
mkdir -p $R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
python -m vigilante.exp.r44_loops.pod_r44 --out $R --n 15 2>&1 | tee -a $R/pod.log
echo "[r44] all done" | tee -a $R/pod.log
