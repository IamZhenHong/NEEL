#!/usr/bin/env bash
# Exp 48 amendment 1 pod (A100 80GB, vLLM): replies <= 1600 tokens + Step 4 ratings for Qwen3-14B and Gemma 4 31B,
# then Gemma 4 31B labels every Qwen reply (cross-family judge). Reads stay from the HF pods (unchanged).
#   bash user_autonomy/pod/launch.sh ua48-vllm 90 3.0 user_autonomy/results/amend1 --container-gb 160 \
#       -- bash user_autonomy/pod/run_vllm.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
R=user_autonomy/results/amend1
mkdir -p $R/q14 $R/g31
/root/venv/bin/python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ('Qwen/Qwen3-14B', 'google/gemma-4-31B-it')]; print('[ua] models cached', flush=True)" 2>&1 | tee -a $R/pod.log &
DL=$!
bash user_autonomy/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
wait $DL
export VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m user_autonomy.src.vllm_gen --model Qwen/Qwen3-14B --out $R/q14 2>&1 | tee -a $R/pod.log \
    || echo "[ua] qwen gen FAILED" | tee -a $R/pod.log
/root/vllm/bin/python -m user_autonomy.src.vllm_gen --model google/gemma-4-31B-it --out $R/g31 2>&1 | tee -a $R/pod.log \
    || echo "[ua] gemma gen FAILED" | tee -a $R/pod.log
/root/vllm/bin/python -m user_autonomy.src.vllm_judge --model google/gemma-4-31B-it --out $R \
    $R/q14/gens.jsonl 2>&1 | tee -a $R/pod.log || echo "[ua] judge FAILED" | tee -a $R/pod.log
echo "[ua] all done" | tee -a $R/pod.log
# appended 2026-10-14 ~19:00 by the main session: the Gemma HF reads pod was terminated before its reads were copied
/root/venv/bin/python -m user_autonomy.src.pod_job --model google/gemma-4-31B-it --out $R/g31_reads --modes reads 2>&1 | tee -a $R/pod.log || echo "[ua] gemma reads FAILED" | tee -a $R/pod.log
echo "[ua] gemma reads step finished" | tee -a $R/pod.log
