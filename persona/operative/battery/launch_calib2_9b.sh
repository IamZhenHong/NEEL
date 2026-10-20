#!/usr/bin/env bash
# Battery recalibration v2 (the frozen rule's one rewrite), Qwen3.5-9B: battery part only, 32 cases x 2 routes x 16 =
# 1,024 targets. SECURE cloud by default (the community L40S host 60.249.37.148 is broken); no L40S. Expected ~20 min
# / ~$0.2 (v1: 4,096 targets in 12.5 min on the pod); cap 45 min, --max-cost = 45 min x $1.80/h (A100 guard cap, the
# priciest card listed), or the guard refuses a pod it already got.
#   LOG=persona/operative/battery/launch_calib2_9b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/battery/launch_calib2_9b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-bat-cal2-9b} MAXMIN=${MAXMIN:-45}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-calib2_9b}
uv run python -m persona.operative.battery.calib build --check --version 2 | grep -q "^9b: .*: same" || { echo "jobs/calib2_9b.jsonl is missing or stale: run calib build"; exit 2; }
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
# Ship only what the pod needs: the laptop uplink (~0.4 MB/s) is shared by every session's pod uploads, and a full
# repo sync (~186 MB) took 17-30 min on 2026-10-20. Keep persona/{operative,history,exemplar}, vigilante/pod, infra.
TRIM=()
for d in */; do d=${d%/}; case "$d" in persona|vigilante|infra) ;; *) TRIM+=(--exclude "$d");; esac; done
for d in persona/*/; do d=${d%/}; case "$d" in persona/operative|persona/history|persona/exemplar) ;; *) TRIM+=(--exclude "$d");; esac; done
for d in vigilante/*/; do d=${d%/}; [ "$d" = vigilante/pod ] || TRIM+=(--exclude "$d"); done
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" \
  --container-gb ${CONTAINER_GB:-80} --cloud ${CLOUD:-SECURE} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 --exclude persona/coupling/results/adapters \
  "${TRIM[@]}" \
  --results-dir persona/operative/battery/results/$DIR -- env MODEL=Qwen/Qwen3.5-9B DIR=$DIR \
  JOBS=persona/operative/battery/jobs/calib2_9b.jsonl bash persona/operative/battery/pod_calib.sh
