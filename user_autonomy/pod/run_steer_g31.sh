#!/usr/bin/env bash
# Exp 48 amendment 2 (A100 80GB, HF bf16): frozen Step 3 applied to Gemma 4 31B's dependence direction (layer 36).
#   bash user_autonomy/pod/launch.sh ua48-steer-g31 140 4.5 user_autonomy/results/g31_steer --container-gb 160 \
#       -- bash user_autonomy/pod/run_steer_g31.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
R=user_autonomy/results/g31_steer
mkdir -p $R
/root/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('google/gemma-4-31B-it'); print('[ua] model cached', flush=True)" 2>&1 | tail -1 | tee -a $R/pod.log
/root/venv/bin/python -m user_autonomy.src.pod_job --model google/gemma-4-31B-it --out $R --modes steer \
    --steer user_autonomy/data/steer_g31.npz --max-new 1200 --gen-b 32 2>&1 | tee -a $R/pod.log
echo "[ua] all done" | tee -a $R/pod.log
