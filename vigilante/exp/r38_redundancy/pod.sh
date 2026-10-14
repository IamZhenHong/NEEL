#!/usr/bin/env bash
# R38 on one A100 80GB: Gemma 4 31B, transformers (project venv).
set -uo pipefail
cd "$(dirname "$0")/../../.."
R=vigilante/exp/r38_redundancy/results
mkdir -p $R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
python -m vigilante.exp.r38_redundancy.pod_r38 --out $R --batch 6 --n 4 2>&1 | tee -a $R/pod.log
echo "[r38] all done" | tee -a $R/pod.log
