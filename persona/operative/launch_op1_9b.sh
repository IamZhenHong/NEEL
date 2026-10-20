#!/usr/bin/env bash
# op1, Qwen3.5-9B: 48 GB cards first, A100 last. Expected ~65 min / ~$0.6 on an A40; cap 100 min, and --max-cost
# covers 100 min on the priciest card listed (A100, $1.80/h guard cap), or the guard refuses a pod it already got.
#   bash persona/operative/launch_op1_9b.sh                       # NAME / MAXMIN / MAXCOST / OP1_GROUPS overridable
#   LOG=persona/operative/launch_op1_9b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/launch_op1_9b.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-op1-9b} MAXMIN=${MAXMIN:-100}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-op1_9b}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" \
  --container-gb ${CONTAINER_GB:-80} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship infra \
  --results-dir persona/operative/results/$DIR -- env MODEL=Qwen/Qwen3.5-9B DIR=$DIR ${OP1_GROUPS:+OP1_GROUPS=$OP1_GROUPS} bash persona/operative/pod_op1.sh
