#!/usr/bin/env bash
# Exp 48 pod 3 (reduced: Step 1 G2 = "style" for both Qwen directions, so no steering per the prereg stop rule):
# Gemma 4 31B (vLLM, greedy) labels every Qwen reply with the primary judge's prompt (cross-family validation).
#   bash user_autonomy/pod/launch.sh ua48-xfam 60 2.0 user_autonomy/results/q14_xfam --container-gb 160 \
#       --sync-extra user_autonomy/results/q14/gens.jsonl -- bash user_autonomy/pod/run_xfam.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
R=user_autonomy/results/q14_xfam
mkdir -p $R
/root/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('google/gemma-4-31B-it'); print('[ua] model cached', flush=True)" 2>&1 | tee -a $R/pod.log &
DL=$!
bash user_autonomy/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
wait $DL
export VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m user_autonomy.src.vllm_judge --model google/gemma-4-31B-it --out $R \
    user_autonomy/results/q14/gens.jsonl 2>&1 | tee -a $R/pod.log
echo "[ua] all done" | tee -a $R/pod.log
