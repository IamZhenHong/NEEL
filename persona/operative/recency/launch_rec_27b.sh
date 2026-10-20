#!/usr/bin/env bash
# Records vs recency, Qwen3.6-27B: 3,456 targets + 384 phase-1 filler replies on an 80 GB A100 (SXM first), 120 GB disk.
# Expected ~45-50 min / ~$1.1-1.3 (op1 27B ~0.42 s/target, longer prompts here, + ~8 min load); cap 80 min,
# --max-cost = 80 min x $1.80/h (A100 guard cap, the priciest card listed).
#   LOG=persona/operative/recency/launch_rec_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/recency/launch_rec_27b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-rec-27b} MAXMIN=${MAXMIN:-80}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-rec_27b}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" \
  --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/recency/results/$DIR -- env MODEL=Qwen/Qwen3.6-27B DIR=$DIR bash persona/operative/recency/pod_rec.sh
