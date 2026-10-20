#!/usr/bin/env bash
# Guarded launcher for a mini-0 jobs file: NAME, MODEL, JOBS, DIR, MAXMIN, MAXCOST from the environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=persona/exemplar
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name ${NAME:?} --allow-concurrent --max-minutes ${MAXMIN:?} --max-cost ${MAXCOST:?} \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb ${CONTAINER_GB:-80} --cloud ${CLOUD:-COMMUNITY} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 \
  --results-dir $E/results/${DIR:?} -- env MODEL=$MODEL JOBS=$JOBS DIR=$DIR MAXLEN=${MAXLEN:-4096} bash $E/pod_jobs.sh
