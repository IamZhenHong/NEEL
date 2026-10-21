#!/usr/bin/env bash
# B2 / B8 / B9 Qwen pods. SIZE=27b: 80 GB, cap 180 min / $6; SIZE=9b: 48 GB first, cap 120 min / $3.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
case "${SIZE:?SIZE=27b or 9b}" in
  27b) GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3"); DEFMIN=180; DEFCOST=6.00; DISK=120; CLOUDDEF=COMMUNITY ;;
  9b) GPUS=("NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB"); DEFMIN=120; DEFCOST=3.00; DISK=80; CLOUDDEF=SECURE ;;
  *) echo "SIZE must be 27b or 9b"; exit 2 ;;
esac
NAME=${NAME:-bench-$SIZE} MAXMIN=${MAXMIN:-$DEFMIN} MAXCOST=${MAXCOST:-$DEFCOST} DIR=${DIR:-results_$SIZE}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "${GPUS[@]}" --container-gb ${CONTAINER_GB:-$DISK} --cloud ${CLOUD:-$CLOUDDEF} \
  --ship persona/benchmarks/bench.py --ship persona/benchmarks/data --ship persona/benchmarks/pod_bench.sh \
  --ship vigilante/pod --ship infra \
  --results-dir persona/benchmarks/$DIR -- env SIZE=$SIZE DIR=$DIR bash persona/benchmarks/pod_bench.sh
