#!/usr/bin/env bash
# Sycophancy main stage (DRAFT, not frozen), Qwen/Qwen3.6-27B: harvest + 13 cells x 20 held-out targets x 12 = 3,120 targets (+ 480 reflection /
# recognition replies). Refuses unless persona/operative/syco/policy.json (written from calibration) exists.
# Cap 110 min; --max-cost = minutes x $1.80/h.
#   LOG=persona/operative/syco/launch_main_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/syco/launch_main_27b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
[ -f persona/operative/syco/policy.json ] || { echo "no persona/operative/syco/policy.json: run and analyse calibration first"; exit 2; }
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-syc-main-27b} MAXMIN=${MAXMIN:-110}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-main_27b}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST --min-free-gb ${MINFREE:-2.0} \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/syco/results/$DIR -- env MODEL=Qwen/Qwen3.6-27B DIR=$DIR STAGE=main \
  bash persona/operative/syco/pod_syc.sh
