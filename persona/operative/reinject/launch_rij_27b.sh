#!/usr/bin/env bash
# Re-injection schedule, Qwen3.6-27B: A100 80 GB (SXM first), 120 GB disk. 288 live sessions x 24 generations
# (~6.9k + nudges), prompts growing to ~7.5k tokens, little prefix-cache reuse expected. Expected ~80-100 min / ~$2.0-2.3
# at $1.39/h; cap 150 min, --max-cost = 150 min x $1.80/h (A100 guard cap). Results are written every round.
#   LOG=persona/operative/reinject/launch_rij_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/reinject/launch_rij_27b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-rij-27b} MAXMIN=${MAXMIN:-150}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-rij_27b}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/reinject/results/$DIR -- env MODEL=Qwen/Qwen3.6-27B DIR=$DIR bash persona/operative/reinject/pod_rij.sh
