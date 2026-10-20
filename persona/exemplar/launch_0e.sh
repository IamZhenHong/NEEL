#!/usr/bin/env bash
# Guarded launcher for mini 0e (separate file so edits never touch a running launch.sh).
set -euo pipefail
cd "$(dirname "$0")/../.."
E=persona/exemplar
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
uv run python infra/launch_run.py --name exm0e --allow-concurrent --max-minutes ${MAXMIN:-45} --max-cost ${MAXCOST:-1.40} \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 80 --cloud ${CLOUD:-COMMUNITY} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 \
  --results-dir $E/results/mini0e -- bash $E/pod_mini0e.sh
