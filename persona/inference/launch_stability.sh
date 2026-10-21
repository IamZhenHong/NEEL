#!/usr/bin/env bash
# Grey-zone sycophancy stability battery launcher, one open model per pod (A100 80GB, 120 GB disk, cap 75 min).
# Ships only the packages the harness imports. --max-cost = minutes x $1.80/h (A100 guard cap). Boot self-destruct
# (launch_run.py >= a57447d) arms at container start; leave RUNPOD_NO_BOOT_SELF_DESTRUCT unset.
#   MODEL=Qwen/Qwen3.6-27B DIR=stab_27b NAME=stab-27b \
#     LOG=persona/inference/launch_stab_27b TRIES=12 WAIT=240 bash persona/history/retry.sh bash persona/inference/launch_stability.sh
#   MODEL=google/gemma-4-31B-it DIR=stab_gemma NAME=stab-gemma ... (same)
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
MODEL=${MODEL:?set MODEL} DIR=${DIR:?set DIR} NAME=${NAME:-stab} GROUPS=${GROUPS:-core}
MAXMIN=${MAXMIN:-75}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --min-free-gb ${MINFREE:-2.0} --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" \
  --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/inference --ship persona/operative --ship persona/history --ship persona/exemplar \
  --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/inference/results/$DIR -- env MODEL=$MODEL DIR=$DIR GROUPS=$GROUPS MODE=${MODE:-full} \
  bash persona/inference/${POD:-pod_stability.sh}
