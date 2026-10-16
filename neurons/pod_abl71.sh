#!/usr/bin/env bash
# e71 label-based ablations (README e71 "Ablation"; D1), inputs neurons/data/abl_inputs.json (build_abl.py).
set -uo pipefail
cd "$(dirname "$0")/.."
R=neurons/results/e71/abl
mkdir -p $R
F="Loading weights\|Fetching\|it/s\]\|falling back\|UserWarning\|warn("
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m neurons.pod_abl --kind qwen36 --out $R --stages e71x e71 spec71 2>&1 | grep --line-buffered -v "$F" | tee -a $R/pod.log
echo "[neurons] abl71 exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
