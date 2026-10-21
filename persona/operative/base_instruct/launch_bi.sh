#!/usr/bin/env bash
# Base vs instruct launcher, PAIR=gemma|qwen9 (base_instruct/DESIGN.md). Ships only the needed paths; the guard refuses
# a card whose rate x MAXMIN exceeds --max-cost (gemma: A100 80GB at <= $1.80/h; qwen9: 48 GB cards at <= $1.00/h).
#   PAIR=gemma bash persona/operative/base_instruct/launch_bi.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
case "${PAIR:?PAIR must be gemma or qwen9}" in
  gemma) M=${MAXMIN:-100}; RATE=1.80; GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); CG=150; CL=${CLOUD:-SECURE} ;;
  qwen9) M=${MAXMIN:-60}; RATE=1.00; GPUS=("NVIDIA A40" "NVIDIA RTX A6000" "NVIDIA L40S"); CG=80; CL=${CLOUD:-SECURE} ;;
  *) echo "PAIR must be gemma or qwen9"; exit 2 ;;
esac
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$M * $RATE / 60:.2f}')")}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
KIND=${KIND:-both}; OUTDIR=${OUTDIR:-$PAIR}
uv run python infra/launch_run.py --name bi-$PAIR-$KIND --allow-concurrent --max-minutes $M --max-cost $MAXCOST \
  --min-free-gb ${MINFREE:-2.0} --gpus "${GPUS[@]}" --container-gb $CG --cloud $CL \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --exclude persona/exemplar/jobs --exclude persona/history/inputs --exclude persona/operative/battery/jobs --exclude persona/operative/reflection_read --exclude persona/operative/syco/jobs --exclude persona/operative/lora_c1e4/data \
  --results-dir persona/operative/base_instruct/results/$OUTDIR -- env PAIR=$PAIR KIND=$KIND OUTDIR=$OUTDIR bash persona/operative/base_instruct/pod_bi.sh
