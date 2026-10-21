#!/usr/bin/env bash
# Tier 2 turn-anchors pod (DESIGN.md), cap 60 min / $2.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-tier2-anchors} MAXMIN=${MAXMIN:-60} MAXCOST=${MAXCOST:-2.00} QG=${QG:-anchors}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" --container-gb 120 --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/reasoning/tier2.py --ship persona/reasoning/q11.py --ship persona/reasoning/q11e.py \
  --ship persona/reasoning/inputs --ship persona/reasoning/pod_tier2.sh \
  --ship persona/operative/op1.py --ship persona/operative/op2/op2.py --ship persona/operative/hardrule/hr.py \
  --ship persona/operative/syco/tasks_syco.py --ship persona/operative/syco/__init__.py --ship persona/inference/__init__.py \
  --ship persona/inference/stability.py --ship persona/inference/stability_items.py \
  --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship infra \
  --results-dir persona/reasoning/${DIR:?} -- env DIR=$DIR QG=$QG bash persona/reasoning/pod_tier2.sh
