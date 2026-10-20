#!/usr/bin/env bash
# rr1 (DESIGN_rr1.md), Qwen3.6-27B on an 80 GB A100 (SXM first), 120 GB disk. ~90 min / ~$2.7; cap 140 min x $1.80/h.
# GPUS_H100=1 adds H100s as a fallback and raises --max-cost to cover them ($3.10/h).
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); RATE=1.80
if [ -n "${GPUS_H100:-}" ]; then GPUS+=("NVIDIA H100 80GB HBM3" "NVIDIA H100 NVL" "NVIDIA H100 PCIe"); RATE=3.10; fi
NAME=${NAME:-rr1-27b} MAXMIN=${MAXMIN:-140}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * $RATE / 60:.2f}')")}
DIR=${DIR:-rr1_27b}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "${GPUS[@]}" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 --exclude persona/coupling/results/adapters \
  --results-dir persona/operative/reflection_read/$DIR -- env MODEL=Qwen/Qwen3.6-27B DIR=$DIR bash persona/operative/reflection_read/pod_rr1.sh
