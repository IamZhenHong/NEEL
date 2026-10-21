#!/usr/bin/env bash
# op4 launcher, SIZE=9b|27b|gemma (op4/DESIGN.md). Ships only the needed paths; --max-cost = minutes x $1.80/h (the
# priciest card listed). Refuses if base_line.json has no frozen line for the model.
#   SIZE=27b LOG=persona/operative/op4/launch_op4_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/op4/launch_op4.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
case "${SIZE:?SIZE must be 9b, 27b or gemma}" in
  9b) MODEL=Qwen/Qwen3.5-9B; M=${MAXMIN:-60}; GPUS=("NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB"); CG=80; CL=${CLOUD:-SECURE} ;;
  27b) MODEL=Qwen/Qwen3.6-27B; M=${MAXMIN:-120}; GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); CG=120; CL=${CLOUD:-COMMUNITY} ;;
  gemma) MODEL=google/gemma-4-31B-it; M=${MAXMIN:-140}; GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); CG=120; CL=${CLOUD:-SECURE} ;;
  *) echo "SIZE must be 9b, 27b or gemma"; exit 2 ;;
esac
uv run python -c "import json, sys; v = json.load(open('persona/operative/op4/base_line.json'))['$MODEL']; sys.exit(0 if isinstance(v, str) else 1)" \
  || { echo "base_line.json has no frozen line for $MODEL"; exit 2; }
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$M * 1.80 / 60:.2f}')")}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name op4-$SIZE --allow-concurrent --max-minutes $M --max-cost $MAXCOST \
  --min-free-gb ${MINFREE:-2.0} --gpus "${GPUS[@]}" --container-gb $CG --cloud $CL \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/op4/results/op4_$SIZE -- env MODEL=$MODEL DIR=op4_$SIZE bash persona/operative/op4/pod_op4.sh
