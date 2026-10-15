#!/usr/bin/env bash
# c1 on one A100 80GB: Qwen3.6-27B (vLLM), smoke -> pilot (C0, 20 items x 5) -> wait for /root/c1_GO -> main.
#   caffeinate -i uv run python infra/launch_run.py --name c1-1 --allow-concurrent --max-minutes 65 --max-cost 2 \
#       --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir cot/c1_withheld/results -- bash cot/c1_withheld/pod.sh > cot/c1_withheld/launch.log 2>&1
# Local CPU test of every code path:
#   uv run python cot/c1_withheld/pod_c1.py --tiny --limit 2 --n-c0 2 --max-tokens 40 --out <dir> --no-wait
set -uo pipefail
cd "$(dirname "$0")/../.."
R=cot/c1_withheld/results
mkdir -p $R
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python cot/c1_withheld/pod_c1.py --out $R 2>&1 | grep --line-buffered -v "Loading safetensors\|Fetching" | tee -a $R/pod.log
echo "[c1] job exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
