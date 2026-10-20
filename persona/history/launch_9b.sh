#!/usr/bin/env bash
# Right-sized launcher for 9B pods (48 GB cards first, A100 last). NAME, MAXMIN, MAXCOST, RDIR, JOB from the environment.
#   NAME=hist1-9b MAXMIN=60 MAXCOST=2.00 RDIR=persona/history/results/batch1_9b \
#   JOB="env MODEL=Qwen/Qwen3.5-9B DIR=batch1_9b bash persona/history/pod_batch1.sh" bash persona/history/launch_9b.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name ${NAME:?} --allow-concurrent --max-minutes ${MAXMIN:?} --max-cost ${MAXCOST:?} \
  --gpus "NVIDIA A40" "NVIDIA L40S" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" \
  --container-gb ${CONTAINER_GB:-80} --cloud ${CLOUD:-COMMUNITY} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 --exclude persona/coupling/results/adapters \
  --results-dir ${RDIR:?} -- ${JOB:?}
