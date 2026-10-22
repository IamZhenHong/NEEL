#!/usr/bin/env bash
# Guarded launcher for A1-A3 (ships only persona/weirdchat without results, vigilante/pod, infra): NAME, MODEL, DIR, MAXMIN,
# MAXCOST from the environment. MAXCOST must cover MAXMIN x rate (the guard refuses otherwise; round up a cent).
set -euo pipefail
cd "$(dirname "$0")/../.."
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
export RUNPOD_ANY_CUDA=${RUNPOD_ANY_CUDA:-1}   # the CUDA filter hides most stock (runpod-workflow lesson)
GPU_LIST=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe")
if [ -n "${GPUS:-}" ]; then IFS=',' read -ra GPU_LIST <<< "$GPUS"; fi
uv run python infra/launch_run.py --name ${NAME:?} --allow-concurrent --max-minutes ${MAXMIN:?} --max-cost ${MAXCOST:?} \
  --gpus "${GPU_LIST[@]}" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-SECURE} --min-free-gb ${MINFREE:-1.5} \
  --ship persona/weirdchat --ship vigilante/pod --ship infra --exclude persona/weirdchat/results \
  ${SYNC_EXTRA:+--sync-extra $SYNC_EXTRA} \
  --results-dir persona/weirdchat/results/${DIR:?} -- env MODEL=$MODEL DIR=$DIR SHARD=${SHARD:-} bash persona/weirdchat/pod_a1.sh
