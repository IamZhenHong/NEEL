#!/usr/bin/env bash
# op1, Qwen3.6-27B on an 80 GB A100 (SXM first; PCIe-first pods have come up with no IP), 120 GB disk.
# Expected ~135 min / ~$3-4 for all groups on one pod; cap 180 min, --max-cost = 180 min x $1.80/h (A100 guard cap).
# PART=a (anc,dyn,think) / PART=b (anc,def,auth,rs,sp) splits it over two pods (~80 min each; anchors run on both).
# GPUS_H100=1 adds H100s as a fallback and raises --max-cost to cover them ($3.10/h).
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
case "${PART:-all}" in
  all) G=""; SUF="" ;;
  a) G="anc,dyn,think"; SUF="_a" ;;
  b) G="anc,def,auth,rs,sp"; SUF="_b" ;;
  *) echo "PART must be all, a or b"; exit 2 ;;
esac
GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); RATE=1.80
if [ -n "${GPUS_H100:-}" ]; then GPUS+=("NVIDIA H100 80GB HBM3" "NVIDIA H100 NVL" "NVIDIA H100 PCIe"); RATE=3.10; fi
NAME=${NAME:-op1-27b$SUF} MAXMIN=${MAXMIN:-$([ -z "$SUF" ] && echo 180 || echo 120)}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * $RATE / 60:.2f}')")}
DIR=${DIR:-op1_27b$SUF}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "${GPUS[@]}" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship infra \
  --results-dir persona/operative/results/$DIR -- env MODEL=Qwen/Qwen3.6-27B DIR=$DIR ${G:+OP1_GROUPS=$G} bash persona/operative/pod_op1.sh
