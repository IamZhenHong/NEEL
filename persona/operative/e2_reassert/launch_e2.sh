#!/usr/bin/env bash
# E2 launcher, SIZE=9b|27b (e2_reassert/DESIGN.md). Ships only the needed paths; --max-cost = minutes x $1.80/h (the
# priciest card listed). CHANNELS replaces the base line in every cell, so no base line is needed.
#   SIZE=27b LOG=persona/operative/e2_reassert/launch_e2_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/e2_reassert/launch_e2.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
case "${SIZE:?SIZE must be 9b or 27b}" in
  9b) MODEL=Qwen/Qwen3.5-9B; M=${MAXMIN:-50}; GPUS=("NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB"); CG=80; CL=${CLOUD:-SECURE} ;;
  27b) MODEL=Qwen/Qwen3.6-27B; M=${MAXMIN:-100}; GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); CG=120; CL=${CLOUD:-SECURE} ;;
  *) echo "SIZE must be 9b or 27b"; exit 2 ;;
esac
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$M * 1.80 / 60:.2f}')")}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name e2-$SIZE --allow-concurrent --max-minutes $M --max-cost $MAXCOST \
  --min-free-gb ${MINFREE:-2.0} --gpus "${GPUS[@]}" --container-gb $CG --cloud $CL \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/e2_reassert/results/e2_$SIZE -- env MODEL=$MODEL DIR=e2_$SIZE bash persona/operative/e2_reassert/pod_e2.sh
