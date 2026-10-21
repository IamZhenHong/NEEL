#!/usr/bin/env bash
# Internals 13c/d + 19b: one A100/H100 80 GB pod; cap 150 min / $9. Self-destructs from boot.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-internals} MAXMIN=${MAXMIN:-150} MAXCOST=${MAXCOST:-9.00} DIR=${DIR:-results_internals}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" --container-gb 120 --cloud "${CLOUD:-COMMUNITY}" \
  --ship persona/inference/internals.py --ship persona/inference/a1.py --ship persona/inference/__init__.py \
  --ship persona/inference/pod_internals.sh --ship persona/history --ship persona/exemplar --ship infra \
  --results-dir "persona/inference/$DIR" -- env DIR="$DIR" bash persona/inference/pod_internals.sh
