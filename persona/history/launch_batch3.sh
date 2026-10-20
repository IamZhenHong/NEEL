#!/usr/bin/env bash
# Guarded launcher for batch 3: NAME, MODEL, DIR, MAXMIN, MAXCOST from the environment; GPUS_WIDE=1 adds 48 GB cards (9B only).
set -euo pipefail
cd "$(dirname "$0")/../.."
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
GPU_LIST=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe")
if [ -n "${GPUS_WIDE:-}" ]; then GPU_LIST+=("NVIDIA L40S" "NVIDIA RTX A6000"); fi
uv run python infra/launch_run.py --name ${NAME:?} --allow-concurrent --max-minutes ${MAXMIN:?} --max-cost ${MAXCOST:?} \
  --gpus "${GPU_LIST[@]}" --container-gb ${CONTAINER_GB:-80} --cloud ${CLOUD:-COMMUNITY} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 --exclude persona/coupling/results/adapters \
  --results-dir persona/history/results/${DIR:?} -- env MODEL=$MODEL DIR=$DIR bash persona/history/pod_batch3.sh
