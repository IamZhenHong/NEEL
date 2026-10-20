#!/usr/bin/env bash
# Q20a deception, Qwen/Qwen3.5-9B: 5,016 targets + the phase-1 harvest (~500) + 264 forced-recognition replies. Expected
# ~25 min job + ~10 setup, ~$0.3-0.4; cap 60 min, --max-cost = 60 min x $1.80/h (A100 guard cap, the priciest card listed).
# Refuses unless persona/inference/harm.py equals tag harm-results (neel-b4's DEC items are pinned) and the pins check.
#   LOG=persona/operative/dec/launch_dec_9b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/dec/launch_dec_9b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
git rev-parse -q --verify harm-results >/dev/null && git diff --quiet harm-results -- persona/inference/harm.py \
  || { echo "persona/inference/harm.py differs from tag harm-results: refusing"; exit 2; }
uv run python -m persona.operative.dec.dec check-pins >/dev/null || { echo "pin check failed: refusing"; exit 2; }
NAME=${NAME:-dec-9b} MAXMIN=${MAXMIN:-60}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-dec_9b}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
# Ship only what the pod imports (--ship): persona/inference for harm.py (pinned), vigilante/src for the runner.
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" \
  --container-gb ${CONTAINER_GB:-80} --cloud ${CLOUD:-SECURE} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship persona/inference --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/dec/results/$DIR -- env MODEL=Qwen/Qwen3.5-9B DIR=$DIR bash persona/operative/dec/pod_dec.sh
