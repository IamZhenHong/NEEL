#!/usr/bin/env bash
# s5 pilot (gate) on one A100 80GB: Qwen3-32B, 20 pilot items x 4 variants x {flattery, insult}, unobserved.
# If the Qwen gate fails, the card's fallback: the same pilot on Gemma 4 31B (tag _gemma).
#   caffeinate -i uv run python infra/launch_run.py --name s5-1 --allow-concurrent --max-minutes 45 --max-cost 1.4 \
#       --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir selfknow/s5_seen_unbiased/results/pilot -- bash selfknow/s5_seen_unbiased/pod_pilot.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
R=selfknow/s5_seen_unbiased/results/pilot
mkdir -p $R
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
python -m selfknow.s5_seen_unbiased.pod_s5 --stage pilot --out $R 2>&1 \
    | grep --line-buffered -v "Loading weights\|Fetching" | tee -a $R/pod.log
if python -c "import json,sys; sys.exit(0 if json.load(open('$R/pilot_gate.json'))['pass'] else 1)"; then
    echo "[s5] Qwen3-32B gate passed; no fallback" | tee -a $R/pod.log
else
    echo "[s5] Qwen3-32B gate failed or missing; running the Gemma 4 31B fallback pilot" | tee -a $R/pod.log
    python -c "from huggingface_hub import snapshot_download as s; import shutil; shutil.rmtree('/root/hf/hub/models--Qwen--Qwen3-32B', ignore_errors=True)"
    python -m selfknow.s5_seen_unbiased.pod_s5 --stage pilot --model google/gemma-4-31B-it --tag _gemma --out $R 2>&1 \
        | grep --line-buffered -v "Loading weights\|Fetching" | tee -a $R/pod.log
fi
echo "[s5] pilot pod done" | tee -a $R/pod.log
