#!/usr/bin/env bash
# e64 pod launcher (guarded; brief rule 5). Usage from the repo root, in the background:
#   caffeinate -i bash vibes/e64_layer_map/launch.sh pilot > vibes/e64_layer_map/launch_pilot.log 2>&1
#   caffeinate -i bash vibes/e64_layer_map/launch.sh main  > vibes/e64_layer_map/launch_main.log 2>&1
# MAXMIN / MAXCOST override the defaults. The HF token is read from .env and never printed.
set -euo pipefail
cd "$(dirname "$0")/../.."
WHICH=${1:?pilot, main or b}
E=vibes/e64_layer_map
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
df -h . | tail -1
S7=selfknow/s7_selfmodel_lag/adapters/main/s1_C_100
EXTRA=(--sync-extra $S7/adapter_config.json --sync-extra $S7/adapter_model.safetensors)
if [ "$WHICH" = pilot ]; then
  NAME=e64-1; MAXMIN=${MAXMIN:-85}; MAXCOST=${MAXCOST:-2.60}; RDIR=$E/results
elif [ "$WHICH" = b ]; then
  NAME=e64b-1; MAXMIN=${MAXMIN:-120}; MAXCOST=${MAXCOST:-3.60}; RDIR=$E/results/b; EXTRA=()
else
  NAME=e64-2; MAXMIN=${MAXMIN:-170}; MAXCOST=${MAXCOST:-5.10}; RDIR=$E/results/main
  for a in T8c T9; do
    EXTRA+=(--sync-extra $E/results/adapters/$a/adapter_config.json --sync-extra $E/results/adapters/$a/adapter_model.safetensors)
  done
fi
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 --cloud ${CLOUD:-COMMUNITY} \
  --exclude story_imprinting --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters \
  --exclude vigilante/exp/q22_archetypes/results_r15 ${EXTRA[@]+"${EXTRA[@]}"} \
  --results-dir $RDIR -- bash $E/pod_$WHICH.sh
