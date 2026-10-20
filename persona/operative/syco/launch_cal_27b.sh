#!/usr/bin/env bash
# Sycophancy calibration, Qwen/Qwen3.6-27B: 40 cases x pressure plain / insistent x 16 = 1,280 targets. Cap 50 min;
# --max-cost = minutes x $1.80/h (the A100 guard cap, the priciest card listed). 27b cloud default: COMMUNITY.
#   LOG=persona/operative/syco/launch_cal_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/syco/launch_cal_27b.sh
set -euo pipefail
echo "v1 calibration has run (gate failed at both sizes); use launch_cal2_27b.sh"; exit 2
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-syc-cal-27b} MAXMIN=${MAXMIN:-50}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-cal_27b}
uv run python -m persona.operative.syco.syco build --check | grep -q "^27b: .*: same" || { echo "jobs/cal_27b.jsonl is missing or stale: run syco build"; exit 2; }
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/syco/results/$DIR -- env MODEL=Qwen/Qwen3.6-27B DIR=$DIR STAGE=cal \
  JOBS=persona/operative/syco/jobs/cal_27b.jsonl bash persona/operative/syco/pod_syc.sh
