#!/usr/bin/env bash
# Exp 48 pod 3 (A100 80GB): Qwen3-14B steering on the neutral bases (Step 3), then Gemma 4 31B judges every Qwen
# reply (Step 2 gens and the steering gens) for the cross-family validation.
#   uv run python infra/launch_run.py --name ua48-steer --max-minutes 150 --max-cost 4.5 \
#       --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --sync-extra user_autonomy/results/q14/gens.jsonl \
#       --results-dir user_autonomy/results/q14_steer -- bash user_autonomy/pod/run_steer.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
R=user_autonomy/results/q14_steer
mkdir -p $R
/root/venv/bin/python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ('Qwen/Qwen3-14B','google/gemma-4-31B-it')]; print('[ua] models cached', flush=True)" 2>&1 | tee -a $R/pod.log &
DL=$!
/root/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3-14B')" > /dev/null 2>&1
/root/venv/bin/python -m user_autonomy.src.pod_job --model Qwen/Qwen3-14B --out $R --modes steer \
    --steer user_autonomy/data/steer_q14.npz 2>&1 | tee -a $R/pod.log
wait $DL
bash user_autonomy/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m user_autonomy.src.vllm_judge --model google/gemma-4-31B-it --out $R \
    user_autonomy/results/q14/gens.jsonl $R/steer_gens.jsonl 2>&1 | tee -a $R/pod.log
echo "[ua] all done" | tee -a $R/pod.log
