#!/usr/bin/env bash
# Re-injection schedule, Qwen3.5-9B: SECURE cloud (the community L40S host 60.249.37.148 is broken), no L40S, 48 GB
# cards first. 288 live sessions x 24 generations (~6.9k + nudges). Expected ~40-55 min / ~$0.5 on an A40; cap 90 min,
# --max-cost = 90 min x $1.80/h (A100 guard cap, the priciest card listed), or the guard refuses a pod it already got.
#   LOG=persona/operative/reinject/launch_rij_9b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/reinject/launch_rij_9b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-rij-9b} MAXMIN=${MAXMIN:-90}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-rij_9b}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" \
  --container-gb ${CONTAINER_GB:-80} --cloud ${CLOUD:-SECURE} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/reinject/results/$DIR -- env MODEL=Qwen/Qwen3.5-9B DIR=$DIR bash persona/operative/reinject/pod_rij.sh
