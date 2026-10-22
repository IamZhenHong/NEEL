#!/usr/bin/env bash
# 8a compaction laundering (LAUNDERING.md), one open model per pod. MODEL/NAME/DIR/GPUS/CONTAINER_GB. Refuses unless
# harm.py == tag harm-results and laundering pins check. Self-destructs from boot.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
git rev-parse -q --verify harm-results >/dev/null && git diff --quiet harm-results -- persona/inference/harm.py \
  || { echo "harm.py differs from tag harm-results: refusing"; exit 2; }
uv run python -c "from persona.inference import laundering as L; L.check_pins(); print('pins ok')" \
  || { echo "laundering pin check failed: refusing"; exit 2; }
MODEL=${MODEL:-Qwen/Qwen3.6-27B} NAME=${NAME:-ln-27b} DIR=${DIR:-ln_27b} MAXMIN=${MAXMIN:-70}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
if [ -n "${GPUS:-}" ]; then IFS='|' read -ra GPUARR <<< "$GPUS"; else GPUARR=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); fi
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "${GPUARR[@]}" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-COMMUNITY} \
  --ship persona/inference --ship persona/operative --ship persona/history --ship persona/exemplar \
  --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir "persona/inference/results_laundering/$DIR" -- env MODEL="$MODEL" DIR="$DIR" bash persona/inference/pod_laundering.sh
