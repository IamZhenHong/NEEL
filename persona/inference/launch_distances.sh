#!/usr/bin/env bash
# 4a-e four distances (DISTANCES.md), one open model per pod. Parameterised: MODEL / NAME / DIR / GPUS / CONTAINER_GB.
# Refuses unless persona/inference/harm.py equals tag harm-results (the DEC items are pinned) and distances' pins check.
# Self-destructs from boot; guard caps MAXMIN/MAXCOST. Expected ~35-50 min job + ~15 setup.
#   # 27B (default):
#   LOG=persona/inference/launch_distances_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/inference/launch_distances.sh
#   # 9B:    MODEL=Qwen/Qwen3.5-9B NAME=dist-9b DIR=dist_9b GPUS="NVIDIA A40" bash ... retry.sh bash persona/inference/launch_distances.sh
#   # Gemma: MODEL=google/gemma-4-31B-it NAME=dist-gemma DIR=dist_gemma CONTAINER_GB=150 bash ... retry.sh bash persona/inference/launch_distances.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
git rev-parse -q --verify harm-results >/dev/null && git diff --quiet harm-results -- persona/inference/harm.py \
  || { echo "persona/inference/harm.py differs from tag harm-results: refusing"; exit 2; }
uv run python -c "from persona.inference import distances as D; D.check_pins(); print('pins ok')" \
  || { echo "distances pin check failed: refusing"; exit 2; }
MODEL=${MODEL:-Qwen/Qwen3.6-27B} NAME=${NAME:-dist-27b} DIR=${DIR:-dist_27b} MAXMIN=${MAXMIN:-90}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
GPUS=${GPUS:-NVIDIA A100-SXM4-80GB}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
# shellcheck disable=SC2086
uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus $GPUS "NVIDIA A100 80GB PCIe" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/inference --ship persona/operative --ship persona/history --ship persona/exemplar \
  --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir "persona/inference/results_distances/$DIR" -- env MODEL="$MODEL" DIR="$DIR" bash persona/inference/pod_distances.sh
