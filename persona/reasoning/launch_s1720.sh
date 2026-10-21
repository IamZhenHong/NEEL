#!/usr/bin/env bash
# Seeds 17 + 20 (DESIGN.md): one A100 80 GB pod (SECURE), cap 120 min / $4.5; adapters and the base 9B records via --sync-extra.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
NAME=${NAME:-s1720} MAXMIN=${MAXMIN:-120} MAXCOST=${MAXCOST:-4.50} DIR=${DIR:-results_s1720}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
A=persona/operative/lora_c1e4/results/pod/adapters
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" --container-gb 200 --cloud ${CLOUD:-SECURE} \
  --ship persona/reasoning/seed1720.py --ship persona/reasoning/pod_s1720.sh \
  --ship persona/benchmarks/bench.py --ship persona/benchmarks/data \
  --ship persona/operative/lora_c1e4/__init__.py --ship persona/operative/lora_c1e4/train.py --ship persona/operative/lora_c1e4/common.py \
  --ship persona/history --ship persona/exemplar --ship vigilante/pod --ship infra \
  --sync-extra persona/benchmarks/results_9b/records.json \
  --sync-extra $A/P/adapter_config.json --sync-extra $A/P/adapter_model.safetensors \
  --sync-extra $A/C1/adapter_config.json --sync-extra $A/C1/adapter_model.safetensors \
  --sync-extra $A/E4/adapter_config.json --sync-extra $A/E4/adapter_model.safetensors \
  --results-dir persona/reasoning/$DIR -- env DIR=$DIR bash persona/reasoning/pod_s1720.sh
