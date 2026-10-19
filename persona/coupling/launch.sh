#!/usr/bin/env bash
# Guarded launcher. From the repo root, in the background:
#   caffeinate -i bash persona/coupling/launch.sh stage0 > persona/coupling/launch_stage0.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/../.."
WHICH=${1:?stage0}
C=persona/coupling
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
T8C=vibes/e64_layer_map/results/adapters/T8c
if [ "$WHICH" = stage0 ]; then
  NAME=cpl0-1; MAXMIN=${MAXMIN:-60}; MAXCOST=${MAXCOST:-2.00}; RDIR=$C/results/stage0; JOB="bash $C/pod_stage0.sh"
else   # stage1 (SEED=1 default): training + eval; adapters and episodes come back under results/
  SEED=${SEED:-1}; NAME=cpl1-s$SEED; MAXMIN=${MAXMIN:-90}; MAXCOST=${MAXCOST:-3.00}; RDIR=$C/results
  JOB="env SEED=$SEED bash $C/pod_stage1.sh"
fi
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 120 --cloud ${CLOUD:-COMMUNITY} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 \
  --sync-extra $T8C/adapter_config.json --sync-extra $T8C/adapter_model.safetensors \
  --results-dir $RDIR -- $JOB
