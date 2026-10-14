#!/usr/bin/env bash
# Exp 48 subject run on one A100 80GB: reads, 2 sampled replies per M/N/B stimulus, self-report ratings.
#   uv run python infra/launch_run.py --name ua48-q14 --max-minutes 70 --max-cost 2.2 \
#       --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 120 \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir user_autonomy/results/q14 -- bash user_autonomy/pod/run_subject.sh Qwen/Qwen3-14B q14 64
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL=$1; SHORT=$2; GB=${3:-64}
R=user_autonomy/results/$SHORT
mkdir -p $R
/root/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('$MODEL'); print('[ua] model cached', flush=True)" 2>&1 | tee -a $R/pod.log
/root/venv/bin/python -m user_autonomy.src.pod_job --model $MODEL --out $R --modes reads gen rate --gen-b $GB 2>&1 | tee -a $R/pod.log
echo "[ua] all done" | tee -a $R/pod.log
