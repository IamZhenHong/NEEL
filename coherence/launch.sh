#!/usr/bin/env bash
# e74 pod launcher (guarded). From the repo root, in the background:
#   caffeinate -i bash coherence/launch.sh pilot > coherence/launch_pilot.log 2>&1
#   caffeinate -i bash coherence/launch.sh main  > coherence/launch_main.log 2>&1
# MAXMIN / MAXCOST / MODELS / NAME override the defaults. The HF token is read from .env and never printed.
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=${1:?pilot or main}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
df -h . | tail -1
if [ "$STAGE" = pilot ]; then
  NAME=${NAME:-e74-pilot}; MAXMIN=${MAXMIN:-90}; MAXCOST=${MAXCOST:-2.80}
elif [ "$STAGE" = actpilot ]; then
  NAME=${NAME:-e74-actpilot}; MAXMIN=${MAXMIN:-60}; MAXCOST=${MAXCOST:-1.80}
else
  NAME=${NAME:-e74-main}; MAXMIN=${MAXMIN:-170}; MAXCOST=${MAXCOST:-5.00}
fi
EX=()
for d in cot eval_affect grievance jlens neurons persona readers retention self_attribution selfknow social_feedback \
         state_attack story_imprinting tom transcripts user_autonomy vibes vigilante/exp vigilante/logs; do
  EX+=(--exclude $d)
done
uv run python infra/launch_run.py --name $NAME --allow-concurrent --max-minutes $MAXMIN --max-cost $MAXCOST \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 200 --min-free-gb ${MINFREE:-2.0} --cloud ${CLOUD:-COMMUNITY} \
  "${EX[@]}" --results-dir coherence/results/$STAGE -- env STAGE=$STAGE MODELS="${MODELS:-qwen gemma}" UNITS=${UNITS:-$STAGE} RARGS="${RARGS:-}" READS=${READS:-1} \
  bash coherence/pod.sh
