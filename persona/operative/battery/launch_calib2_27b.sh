#!/usr/bin/env bash
# Battery recalibration v2 (the frozen rule's one rewrite), Qwen3.6-27B: battery part only, 1,024 targets, on an 80 GB
# A100 (SXM first; PCIe-first pods have come up with no IP), 120 GB disk. Expected ~30 min / ~$0.7 (v1: 4,096 targets
# in 21.5 min on the pod); cap 60 min, --max-cost = 60 min x $1.80/h (A100 guard cap).
#   LOG=persona/operative/battery/launch_calib2_27b TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/battery/launch_calib2_27b.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-bat-cal2-27b} MAXMIN=${MAXMIN:-60}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
DIR=${DIR:-calib2_27b}
uv run python -m persona.operative.battery.calib build --check --version 2 | grep -q "^27b: .*: same" || { echo "jobs/calib2_27b.jsonl is missing or stale: run calib build"; exit 2; }
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
# Ship only what the pod imports (--ship; the laptop uplink is shared by every session). vigilante/src is
# needed: persona.history.runner imports vigilante.src.templates / .sim (a dry run without it fails at import).
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir persona/operative/battery/results/$DIR -- env MODEL=Qwen/Qwen3.6-27B DIR=$DIR \
  JOBS=persona/operative/battery/jobs/calib2_27b.jsonl bash persona/operative/battery/pod_calib.sh
