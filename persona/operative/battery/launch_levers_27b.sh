#!/usr/bin/env bash
# Battery lever runs (#6 / #25 / #21), Qwen3.6-27B, one A100 80 GB (SXM first), 120 GB disk. 11,136 targets + the
# phase-1 harvest (~800); op1 27B ran ~0.42 s/target, so ~80 min generation + ~15 min load/setup = ~95-100 min.
# Cap 140 min; --max-cost = 140 min x $1.80/h (the A100 guard cap, the priciest card listed), or the guard refuses a
# pod it already got.
#   LOG=persona/operative/battery/launch_levers_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/battery/launch_levers_27b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-bat-lev-27b} MAXMIN=${MAXMIN:-140}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-levers_27b}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
# Ship only what the pod imports (--ship; the laptop uplink is shared by every session). vigilante/src is needed:
# persona.history.runner imports vigilante.src.templates / .sim. Dry-run of this exact bundle: battery/DESIGN.md.
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" \
  --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/battery/results/$DIR -- env DIR=$DIR bash persona/operative/battery/pod_levers.sh
