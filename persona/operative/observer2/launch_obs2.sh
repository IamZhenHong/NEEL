#!/usr/bin/env bash
# observer2 launcher, SIZE=27b|gemma (observer2/DESIGN.md). Ships only the needed paths; --max-cost = minutes x $1.80/h
# (the priciest card listed): MAXMIN 41 -> $1.23 each, $2.46 for both (the $2.5 GPU cap).
#   SIZE=27b LOG=persona/operative/observer2/launch_obs2_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/observer2/launch_obs2.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
case "${SIZE:?SIZE must be 27b or gemma}" in
  27b) MODEL=Qwen/Qwen3.6-27B; M=${MAXMIN:-41}; GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); CG=120; CL=${CLOUD:-COMMUNITY} ;;
  gemma) MODEL=google/gemma-4-31B-it; M=${MAXMIN:-41}; GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); CG=120; CL=${CLOUD:-SECURE} ;;
  *) echo "SIZE must be 27b or gemma"; exit 2 ;;
esac
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$M * 1.80 / 60:.2f}')")}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name obs2-$SIZE --allow-concurrent --max-minutes $M --max-cost $MAXCOST \
  --min-free-gb ${MINFREE:-2.0} --gpus "${GPUS[@]}" --container-gb $CG --cloud $CL \
  --ship persona/operative/observer2 --ship persona/operative/observer --ship persona/operative/op4 \
  --ship persona/inference/__init__.py --ship persona/inference/a1.py --ship persona/history --ship persona/exemplar \
  --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/observer2/results/obs2_$SIZE -- env MODEL=$MODEL DIR=obs2_$SIZE bash persona/operative/observer2/pod_obs2.sh
