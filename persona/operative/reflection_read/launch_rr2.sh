#!/usr/bin/env bash
# rr2 (DESIGN_rr2.md). SIZE=27b: A100 80 GB (SXM first), cap 75 min; SIZE=9b: 48 GB cards first, A100 last, cap 60 min.
# --max-cost covers the cap on the priciest card listed ($1.80/h A100 guard cap). Uploads only what the pod imports.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
case "${SIZE:?SIZE=27b or 9b}" in
  27b) MODEL=Qwen/Qwen3.6-27B; GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); DEFMIN=75; DISK=120 ;;
  9b) MODEL=Qwen/Qwen3.5-9B; GPUS=("NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB"); DEFMIN=60; DISK=80 ;;
  *) echo "SIZE must be 27b or 9b"; exit 2 ;;
esac
NAME=${NAME:-rr2-$SIZE} MAXMIN=${MAXMIN:-$DEFMIN}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-rr2_$SIZE}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "${GPUS[@]}" --container-gb ${CONTAINER_GB:-$DISK} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship infra \
  --exclude persona/operative/reflection_read/rr1_27b --exclude persona/operative/battery \
  --results-dir persona/operative/reflection_read/$DIR -- env MODEL=$MODEL DIR=$DIR bash persona/operative/reflection_read/pod_rr2.sh
