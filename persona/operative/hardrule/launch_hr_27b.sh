#!/usr/bin/env bash
# Mini 3, Qwen3.6-27B: 3,264 targets on an 80 GB A100 (SXM first), 120 GB disk. Expected ~40-45 min / ~$1.0-1.3
# (op1 27B ~0.42 s/target + ~8 min load); cap 75 min, --max-cost = 75 min x $1.80/h (A100 guard cap).
#   LOG=persona/operative/hardrule/launch_hr_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/hardrule/launch_hr_27b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-hr-27b} MAXMIN=${MAXMIN:-75}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-hr_27b}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" \
  --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/hardrule/results/$DIR -- env MODEL=Qwen/Qwen3.6-27B DIR=$DIR bash persona/operative/hardrule/pod_hr.sh
