#!/usr/bin/env bash
# jlens on one A100 80GB: Qwen3.6-27B with the Neuronpedia J-lens / J++ lens; self-check, gate, then the claim reads.
#   uv run python infra/launch_run.py --name jlens-1 --max-minutes 75 --max-cost 4 \
#       --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir jlens/results -- bash jlens/pod.sh
# Local CPU test of every code path: uv run python -m jlens.pod_jlens --tiny --limit 8 --out <dir>
set -uo pipefail
cd "$(dirname "$0")/.."
R=jlens/results
mkdir -p $R
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
python -m jlens.pod_jlens --out $R 2>&1 | grep --line-buffered -v "Loading weights\|Fetching" | tee -a $R/pod.log
echo "[jlens] job exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
