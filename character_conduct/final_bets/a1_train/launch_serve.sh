#!/usr/bin/env bash
# CC-F-A1 pod S launcher: one guarded community A100 80GB (pod_serve.sh: stage 1 generation, then the evaluation
# server), booked to character_conduct/ledger.jsonl as CC-F-A1.
#   bash character_conduct/final_bets/a1_train/launch_serve.sh        # MAXMIN 1200 at <= $1.39/h: MAXCOST $27.80
# infra/launch_run.py is called directly; once "[pod] running (detached)" prints, the launcher is killed by hand
# (detached run: the pod's boot self-destruct at MAXMIN + 30 min is the backstop), results are pulled over ssh and the
# pod is terminated and booked by hand (elapsed x rate). The laptop reaches the server with
#   ssh -N -L 8000:127.0.0.1:8000 -p PORT root@HOST     (PORT / HOST from the "[pod] ssh" line of the launch log)
# Refuses unless the tag cc-f-a1-prereg exists and LG.check_both passes for CC-F-A1 (RunPod cap $45) with the bound.
# No HF token is passed (Qwen3-14B is public). A1_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
DRY=${A1_LAUNCH_DRY:-}
export RUNPOD_ANY_CUDA=1
unset HF_TOKEN
NAME=cc-f-a1-serve; RES=$H/results/serve; MAXMIN=${MAXMIN:-1200}; RATE=${A1_RATE:-139}
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
for p in character_conduct/__init__.py character_conduct/final_bets/__init__.py "$H" \
         self_concept/__init__.py self_concept/n4_conditional/__init__.py self_concept/n4_conditional/score.py \
         self_concept/n4_conditional/tasks.py self_concept/n4_conditional/data/tasks.jsonl \
         self_concept/n4_conditional/data/task_meta.json vigilante/pod/setup_vllm.sh; do SHIP+=(--ship "$p"); done
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150 "${SHIP[@]}"
     --exclude 'data/adapters*' --exclude "$H/results" --exclude '__pycache__' --hf-token ""
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_serve.sh")
if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
git rev-parse -q --verify "refs/tags/cc-f-a1-prereg" > /dev/null 2>&1 || { echo "[prereg] no tag cc-f-a1-prereg"; exit 2; }
BOUND=$(python3 -c "import math, sys; print(f'{math.ceil((int(sys.argv[1]) + 30) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
uv run python -c "
import sys; import character_conduct.common.ledger as LG
LG.check_both('CC-F-A1', 45.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: the cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
trap - INT TERM HUP
exit "$rc"
