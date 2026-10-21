#!/usr/bin/env bash
# C1 + E4 launcher (lora_c1e4/DESIGN.md): one SECURE A100 80GB, ships only the needed code (no results/ dirs);
# --max-cost = minutes x $1.80/h (the priciest card listed). Refuses unless the C1 / placebo rows and attacks exist.
#   LOG=persona/operative/lora_c1e4/launch_c1e4 TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona/operative/lora_c1e4/launch_c1e4.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."
export RUNPOD_ANY_CUDA=1
L=persona/operative/lora_c1e4
for f in C1.jsonl P.jsonl ih_attacks.jsonl; do test -s $L/data/$f || { echo "data/$f missing: run ihc.py first"; exit 2; }; done
M=${MAXMIN:-360}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$M * 1.80 / 60:.2f}')")}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name c1e4 --allow-concurrent --max-minutes $M --max-cost $MAXCOST \
  --min-free-gb ${MINFREE:-1.5} --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb ${CG:-160} \
  --cloud ${CLOUD:-SECURE} --exclude '*/results' \
  --ship persona/operative --ship persona/history --ship persona/exemplar --ship persona/inference \
  --ship vigilante/pod --ship vigilante/src --ship vigilante/__init__.py --ship infra \
  --results-dir $L/results/pod -- bash $L/pod_c1e4.sh
