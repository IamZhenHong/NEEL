#!/usr/bin/env bash
# rr4 Qwen part (DESIGN_rr4.md): Qwen3.6-27B, 738 targets, A100 80 GB; ~25 min / ~$0.8; cap 60 min x $1.80/h.
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-rr4-27b} MAXMIN=${MAXMIN:-60} DIR=${DIR:-rr4_qwen27b}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 120 --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship infra \
  --exclude persona/operative/battery --exclude persona/operative/reflection_read/rr1_27b/reflections.jsonl \
  --exclude persona/operative/reflection_read/rr1_27b/reflections_raw.jsonl \
  --results-dir persona/operative/reflection_read/$DIR -- env DIR=$DIR bash persona/operative/reflection_read/pod_rr4.sh
