#!/usr/bin/env bash
# Guarded launcher for mini 0h (separate file so edits never touch a running launch.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
E=persona/exemplar
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
GPU_LIST=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe")
if [ -n "${GPUS_WIDE:-}" ]; then GPU_LIST+=("NVIDIA L40S" "NVIDIA RTX A6000"); fi   # 9B fits on 48 GB
uv run python infra/launch_run.py --name exm0h --allow-concurrent --max-minutes ${MAXMIN:-60} --max-cost ${MAXCOST:-2.50} \
  --gpus "${GPU_LIST[@]}" --container-gb 80 --cloud ${CLOUD:-COMMUNITY} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 \
  --results-dir $E/results/mini0h -- bash $E/pod_mini0h.sh
