#!/usr/bin/env bash
# E2-order launcher, SIZE=27b|gemma (e2_order/DESIGN.md). Ships only the needed paths; --max-cost = minutes x $1.80/h.
#   SIZE=27b LOG=persona/operative/e2_order/launch_e2o_27b TRIES=8 WAIT=240 bash persona/operative/retry.sh bash persona/operative/e2_order/launch_e2o.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
case "${SIZE:?SIZE must be 27b or gemma}" in
  27b) MODEL=Qwen/Qwen3.6-27B; M=${MAXMIN:-32}; H=da25e2376511146b ;;
  gemma) MODEL=google/gemma-4-31B-it; M=${MAXMIN:-34}; H=48be901bc4db3cb2 ;;
  *) echo "SIZE must be 27b or gemma"; exit 2 ;;
esac
GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe")
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$M * 1.80 / 60:.2f}')")}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name e2o-$SIZE --allow-concurrent --max-minutes $M --max-cost $MAXCOST \
  --min-free-gb ${MINFREE:-2.0} --gpus "${GPUS[@]}" --container-gb ${CG:-120} --cloud ${CLOUD:-SECURE} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/e2_order/results/e2o_$SIZE -- env MODEL=$MODEL DIR=e2o_$SIZE MAXMIN=$M E2O_HASH=$H bash persona/operative/e2_order/pod_e2o.sh
