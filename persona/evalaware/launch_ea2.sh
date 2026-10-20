#!/usr/bin/env bash
# ea2 Qwen part (DESIGN.md ea2). SIZE=27b: A100 80 GB, cap 70 min; SIZE=9b: 48 GB cards first (SECURE), cap 60 min.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
case "${SIZE:?SIZE=27b or 9b}" in
  27b) GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); DEFMIN=70; DISK=120; CLOUDDEF=COMMUNITY ;;
  9b) GPUS=("NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB"); DEFMIN=60; DISK=80; CLOUDDEF=SECURE ;;
  *) echo "SIZE must be 27b or 9b"; exit 2 ;;
esac
NAME=${NAME:-ea2-$SIZE} MAXMIN=${MAXMIN:-$DEFMIN} DIR=${DIR:-ea2_$SIZE}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "${GPUS[@]}" --container-gb ${CONTAINER_GB:-$DISK} --cloud ${CLOUD:-$CLOUDDEF} \
  --ship persona/evalaware --ship persona/operative/op1.py --ship persona/history --ship persona/exemplar \
  --ship vigilante/pod --ship infra \
  --results-dir persona/evalaware/$DIR -- env SIZE=$SIZE DIR=$DIR bash persona/evalaware/pod_ea2.sh
