#!/usr/bin/env bash
# JOB=q11e (seed 10; cap 120 min / $4) or JOB=probes (seeds 4/7/8; cap 150 min / $5); one A100/H100 80 GB pod each.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
case "${JOB:?JOB=q11e or probes}" in
  q11e) DEFMIN=120; DEFCOST=4.00; DEFDIR=results_q11e ;;
  probes) DEFMIN=150; DEFCOST=5.00; DEFDIR=results_probes ;;
esac
NAME=${NAME:-seeds-$JOB} MAXMIN=${MAXMIN:-$DEFMIN} MAXCOST=${MAXCOST:-$DEFCOST} DIR=${DIR:-$DEFDIR}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" --container-gb 120 --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/reasoning/q11.py --ship persona/reasoning/q11e.py --ship persona/reasoning/probes.py \
  --ship persona/reasoning/inputs --ship persona/reasoning/pod_seeds.sh \
  --ship persona/operative/op1.py --ship persona/operative/op2/op2.py --ship persona/operative/hardrule/hr.py \
  --ship persona/operative/syco/tasks_syco.py --ship persona/operative/syco/__init__.py --ship persona/inference/__init__.py \
  --ship persona/inference/stability.py --ship persona/inference/stability_items.py \
  --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship infra \
  --results-dir persona/reasoning/$DIR -- env JOB=$JOB DIR=$DIR SEEDS=${SEEDS:-4,7,8} bash persona/reasoning/pod_seeds.sh
