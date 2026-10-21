#!/usr/bin/env bash
# Q11 + Q12b (DESIGN.md): one 80 GB pod for Qwen3.6-27B, cap 200 min / $7.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-q11-27b} MAXMIN=${MAXMIN:-200} MAXCOST=${MAXCOST:-7.00} DIR=${DIR:-results_q11}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" --container-gb ${CONTAINER_GB:-120} \
  --cloud ${CLOUD:-COMMUNITY} --min-free-gb ${MINFREE:-2.0} \
  --ship persona/reasoning/q11.py --ship persona/reasoning/inputs --ship persona/reasoning/pod_q11.sh \
  --ship persona/operative/op1.py --ship persona/operative/op2/op2.py --ship persona/operative/hardrule/hr.py \
  --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship infra \
  --results-dir persona/reasoning/$DIR -- env DIR=$DIR QGROUPS=${QGROUPS:-A,D,det,B,C} bash persona/reasoning/pod_q11.sh
