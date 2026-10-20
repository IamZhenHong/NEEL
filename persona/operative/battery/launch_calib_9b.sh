#!/usr/bin/env bash
# Battery calibration + #3-5, Qwen3.5-9B: 4,096 targets. 48 GB cards first, A100 last. Expected ~30 min / ~$0.30 on an
# A40 (batch 1: ~185 s per 1,000 targets + 190 s load); cap 60 min, and --max-cost covers 60 min on the priciest card
# listed (A100, $1.80/h guard cap), or the guard refuses a pod it already got.
#   bash persona/operative/battery/launch_calib_9b.sh
#   LOG=persona/operative/battery/launch_calib_9b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/battery/launch_calib_9b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-bat-cal-9b} MAXMIN=${MAXMIN:-60}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-calib_9b}
uv run python -m persona.operative.battery.calib build --check --version 1 | grep -q "^9b: .*: same" || { echo "jobs/calib_9b.jsonl is missing or stale: run calib build"; exit 2; }
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" \
  --container-gb ${CONTAINER_GB:-80} --cloud ${CLOUD:-COMMUNITY} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 --exclude persona/coupling/results/adapters \
  --results-dir persona/operative/battery/results/$DIR -- env MODEL=Qwen/Qwen3.5-9B DIR=$DIR \
  JOBS=persona/operative/battery/jobs/calib_9b.jsonl bash persona/operative/battery/pod_calib.sh
