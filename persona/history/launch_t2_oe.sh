#!/usr/bin/env bash
# Guarded launcher for T2-OE (open-ended actions) (ships only the needed paths): NAME, MODEL, DIR, MAXMIN, MAXCOST from
# the environment. Secure A100 80 GB at $1.79/hr: 27B MAXMIN 55 / MAXCOST 1.65, Gemma MAXMIN 38 / MAXCOST 1.14.
set -euo pipefail
cd "$(dirname "$0")/../.."
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
export RUNPOD_ANY_CUDA=${RUNPOD_ANY_CUDA:-1}   # the CUDA filter hides most stock (runpod-workflow lesson)
GPU_LIST=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe")
if [ -n "${GPUS:-}" ]; then IFS=',' read -ra GPU_LIST <<< "$GPUS"; fi   # e.g. GPUS="NVIDIA A100 80GB PCIe"
uv run python infra/launch_run.py --name ${NAME:?} --allow-concurrent --max-minutes ${MAXMIN:?} --max-cost ${MAXCOST:?} \
  --gpus "${GPU_LIST[@]}" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-SECURE} \
  --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py \
  --ship infra --exclude persona/exemplar/jobs --results-dir persona/history/results/${DIR:?} \
  -- env MODEL=$MODEL DIR=$DIR MAXMIN=$MAXMIN bash persona/history/pod_t2_oe.sh
