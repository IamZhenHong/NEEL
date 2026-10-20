#!/usr/bin/env bash
# persona/inference P1-b launcher. SIZE=9b: 48 GB cards first (A40, A6000; no L40S: broken-host history), A100 last,
# cap 75 min. SIZE=27b: A100 SXM first, 120 GB disk, cap 120 min. --max-cost = cap x $1.80/h (A100 guard cap).
#   SIZE=9b LOG=persona/inference/launch_9b TRIES=12 WAIT=240 bash persona/history/retry.sh bash persona/inference/launch_infer.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
case "${SIZE:?}" in
  9b) MODEL=Qwen/Qwen3.5-9B; GPUS=("NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB"); DISK=80; MAXMIN=${MAXMIN:-75} ;;
  27b) MODEL=Qwen/Qwen3.6-27B; GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); DISK=120; MAXMIN=${MAXMIN:-120} ;;
  *) echo "SIZE must be 9b or 27b"; exit 2 ;;
esac
POD=${POD:-pod_infer.sh} NAME=${NAME:-inf-$SIZE} DIR=${DIR:-inf_$SIZE}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
# TRIM: the shared laptop uplink is ~0.4 MB/s; a full repo sync is ~186 MB (30+ min, which eats the time cap counted
# from pod creation). The pod imports only persona/ and vigilante/, so exclude every other top-level dir (neel-b3,
# 2026-10-20). This drops the sync to a few MB. `results/` etc. are already excluded by launch_run.py.
TRIM=()
for d in coherence cot eval_affect grievance infra jlens mas_pilot neurons olmo_persona prep readers \
         rejection_values retention self_attribution selfknow social_feedback state_attack story_imprinting \
         tom user_autonomy vibes welfare_training; do TRIM+=(--exclude "$d"); done
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "${GPUS[@]}" --container-gb ${CONTAINER_GB:-$DISK} --cloud ${CLOUD:-COMMUNITY} \
  "${TRIM[@]}" --exclude vigilante/exp --exclude persona/coupling/results/adapters \
  --results-dir persona/inference/results/$DIR -- env MODEL=$MODEL DIR=$DIR bash persona/inference/$POD
