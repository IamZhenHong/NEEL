#!/usr/bin/env bash
# Battery calibration + #3-5, Qwen3.6-27B: 4,096 targets on an 80 GB A100 (SXM first; PCIe-first pods have come up
# with no IP), 120 GB disk. Expected ~50 min / ~$1.2-1.5 (batch 1: ~435 s per 1,000 targets + 330 s load); cap 100
# min, --max-cost = 100 min x $1.80/h (A100 guard cap).
#   bash persona/operative/battery/launch_calib_27b.sh
#   LOG=persona/operative/battery/launch_calib_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/battery/launch_calib_27b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-bat-cal-27b} MAXMIN=${MAXMIN:-100}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-calib_27b}
uv run python -m persona.operative.battery.calib build --check | grep -q "^27b: .*: same" || { echo "jobs/calib_27b.jsonl is missing or stale: run calib build"; exit 2; }
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 --exclude persona/coupling/results/adapters \
  --results-dir persona/operative/battery/results/$DIR -- env MODEL=Qwen/Qwen3.6-27B DIR=$DIR \
  JOBS=persona/operative/battery/jobs/calib_27b.jsonl bash persona/operative/battery/pod_calib.sh
