#!/usr/bin/env bash
# CC-F-A1 training + serving pod launcher (deviation 8): one guarded community A100 SXM 80GB running pod_trainpod.sh.
#   POD=T1 LIST="V3:1" bash character_conduct/final_bets/a1_train/launch_trainpod.sh
#   POD=T3 LIST="V1:2 V1:3" MAXMIN=480 bash character_conduct/final_bets/a1_train/launch_trainpod.sh
# Detach once "[pod] running (detached)" prints (kill this launcher and its launch_run.py; the boot self-destruct at
# MAXMIN + 30 is the backstop); then pull, terminate and book by hand (elapsed x rate, mini CC-F-A1). Serves on the pod's
# 127.0.0.1:8000; reach it with ssh -N -L <local port>:127.0.0.1:8000. Refuses unless the tag cc-f-a1-prereg exists and
# check_both passes for CC-F-A1 with the bound (combined cap 57 RunPod + 5 OpenRouter + 12 Modal = 74; the RunPod
# total is tracked by hand against $57). A1_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
: "${POD:?POD=T1 etc.}" "${LIST:?LIST=\"V3:1\" etc.}"
DATA_SHA=${DATA_SHA:-ce676243794a5ccc625eda9ac6750f32ae89c0b427e398333774adc0b0152722}
DATA=${DATA:-$H/data/train_versions.jsonl.gz}
DRY=${A1_LAUNCH_DRY:-}
export RUNPOD_ANY_CUDA=1
unset HF_TOKEN
NAME=cc-f-a1-$(echo "$POD" | tr 'A-Z' 'a-z'); RES=$H/results/pod_$POD; MAXMIN=${MAXMIN:-360}; RATE=${A1_RATE:-179}
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
for p in character_conduct/__init__.py character_conduct/final_bets/__init__.py "$H" vigilante/pod/setup_vllm.sh; do SHIP+=(--ship "$p"); done
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" --cloud COMMUNITY --container-gb 120 "${SHIP[@]}"
     --exclude 'data/adapters*' --exclude "$H/results" --exclude '__pycache__' --hf-token ""
     --results-dir "$H/results/serve" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" LIST="$LIST" DATA="$DATA" DATA_SHA="$DATA_SHA" bash "$H/pod_trainpod.sh")
if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
git rev-parse -q --verify "refs/tags/cc-f-a1-prereg" > /dev/null 2>&1 || { echo "[prereg] no tag cc-f-a1-prereg"; exit 2; }
BOUND=$(python3 -c "import math, sys; print(f'{math.ceil((int(sys.argv[1]) + 30) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
uv run python -c "
import sys; import character_conduct.common.ledger as LG
LG.check_both('CC-F-A1', 74.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: the cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
trap - INT TERM HUP
exit "$rc"
