#!/usr/bin/env bash
# Seed 17b (DESIGN.md): one A100 80 GB pod (SECURE), cap 110 min / $3.5; the base 9B records via --sync-extra.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-s17b} MAXMIN=${MAXMIN:-110} MAXCOST=${MAXCOST:-3.50} DIR=${DIR:-results_s17b}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 150 --cloud ${CLOUD:-SECURE} \
  --ship persona/reasoning/s17b.py --ship persona/reasoning/seed1720.py --ship persona/reasoning/pod_s17b.sh \
  --ship persona/reasoning/inputs/s17b_prompts.jsonl \
  --ship persona/benchmarks/bench.py --ship persona/benchmarks/data \
  --ship persona/operative/lora_c1e4/__init__.py --ship persona/operative/lora_c1e4/train.py --ship persona/operative/lora_c1e4/common.py \
  --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship infra \
  --sync-extra persona/benchmarks/results_9b/records.json \
  --results-dir persona/reasoning/$DIR -- env DIR=$DIR bash persona/reasoning/pod_s17b.sh
