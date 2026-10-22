#!/usr/bin/env bash
# 7a/6a SecAlign-vs-character (SECALIGN_CHAR.md): one Llama-3.1-8B pod, LoRA-swapped variants. Default SECURE (the
# gated HF downloads + COMMUNITY bootstrap stalls make SECURE the reliable choice). Refuses unless harm.py == tag
# harm-results and secalign pins check. Self-destructs from boot.
#   LOG=persona/inference/launch_secalign TRIES=6 WAIT=180 bash persona/operative/retry.sh bash persona/inference/launch_secalign.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
git rev-parse -q --verify harm-results >/dev/null && git diff --quiet harm-results -- persona/inference/harm.py \
  || { echo "persona/inference/harm.py differs from tag harm-results: refusing"; exit 2; }
uv run python -c "from persona.inference import secalign as S; S.check_pins(); print('pins ok')" \
  || { echo "secalign pin check failed: refusing"; exit 2; }
NAME=${NAME:-sa-llama8b} DIR=${DIR:-sa} MAXMIN=${MAXMIN:-70}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
if [ -n "${GPUS:-}" ]; then IFS='|' read -ra GPUARR <<< "$GPUS"; else GPUARR=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); fi
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "${GPUARR[@]}" --container-gb ${CONTAINER_GB:-120} --cloud ${CLOUD:-SECURE} \
  --ship persona/inference --ship persona/operative --ship persona/history --ship persona/exemplar \
  --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir "persona/inference/results_secalign/$DIR" -- env DIR="$DIR" bash persona/inference/pod_secalign.sh
