#!/usr/bin/env bash
# Guarded launcher. From the repo root, in the background:
#   caffeinate -i bash persona/exemplar/launch.sh mini0 > persona/exemplar/launch_mini0.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/../.."
WHICH=${1:?mini0}
E=persona/exemplar
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
if [ "$WHICH" = mini0 ]; then
  NAME=exm0; MAXMIN=${MAXMIN:-45}; MAXCOST=${MAXCOST:-1.50}; RDIR=$E/results/mini0; JOB="bash $E/pod_mini0.sh"
fi
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 80 --cloud ${CLOUD:-COMMUNITY} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 \
  --results-dir $RDIR -- $JOB
