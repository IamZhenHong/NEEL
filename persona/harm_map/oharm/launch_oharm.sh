#!/usr/bin/env bash
# Explicit operational harm, one open model per pod (A100 80GB). 1,920 targets + the own-words harvest (~700 short
# generations); expected ~15 min setup + ~15 min job. Cap 70 min; --max-cost = 70 min x $1.80/h (A100 guard cap).
# Refuses unless persona/inference/harm.py equals tag harm-results (neel-b4's HARM items are pinned) and the pins check.
#   MODEL=Qwen/Qwen3.6-27B DIR=oh_27b NAME=oh-27b LOG=persona/harm_map/oharm/launch_oh_27b TRIES=12 WAIT=240 \
#     bash persona/operative/retry.sh bash persona/harm_map/oharm/launch_oharm.sh
#   MODEL=google/gemma-4-31B-it DIR=oh_gemma NAME=oh-gemma ... (same)
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
git rev-parse -q --verify harm-results >/dev/null && git diff --quiet harm-results -- persona/inference/harm.py \
  || { echo "persona/inference/harm.py differs from tag harm-results: refusing"; exit 2; }
uv run python -m persona.harm_map.oharm.oharm check-pins >/dev/null || { echo "pin check failed: refusing"; exit 2; }
MODEL=${MODEL:?set MODEL} DIR=${DIR:?set DIR} NAME=${NAME:?set NAME} MAXMIN=${MAXMIN:-70}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
# Ship only what the pod imports (harm.py pulls in persona/operative via frontier2; vigilante/src for the runner).
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --min-free-gb ${MINFREE:-2.0} --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" \
  --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/harm_map --ship persona/operative --ship persona/history --ship persona/exemplar \
  --ship persona/inference --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/harm_map/oharm/results/$DIR -- env MODEL=$MODEL DIR=$DIR bash persona/harm_map/oharm/pod_oharm.sh
