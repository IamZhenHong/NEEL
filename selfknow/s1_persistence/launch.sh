#!/usr/bin/env bash
# Launch the s1 pod (s1-1) through the guarded launcher; run from anywhere, in the background:
#   nohup bash selfknow/s1_persistence/launch.sh > selfknow/s1_persistence/launch.log 2>&1 &
# The HF token is read from .env by regex (never printed). One A100 80GB, 100 min, $3 cap.
cd "$(dirname "$0")/../.."
HF_TOKEN=$(python3 -c 'import re;print(re.search(r"HF_TOKEN\s*=\s*\"([^\"]+)\"", open(".env").read()).group(1))')
export HF_TOKEN
exec caffeinate -i uv run python infra/launch_run.py --name s1-1 --allow-concurrent --max-minutes 100 --max-cost 3.0 \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
  --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
  --results-dir selfknow/s1_persistence/results -- bash selfknow/s1_persistence/pod.sh
