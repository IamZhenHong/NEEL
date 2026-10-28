#!/usr/bin/env bash
# CC-F-A1 pod T launcher: one guarded community pod with two A100 80GB (pod_train.sh), booked to
# character_conduct/ledger.jsonl as CC-F-A1.
#   bash character_conduct/final_bets/a1_train/launch_train.sh        # MAXMIN 240 x 2 GPUs at <= $1.39/h: MAXCOST $11.12
# Needs data/train_versions.jsonl.gz (data_a1.py build; label check passed) and its sha256 in DATA_SHA. Detached as pod
# S: once "[pod] running (detached)" prints, kill the launcher; copy adapters to pod S after each ROUNDn_DONE; touch
# /root/COPIED; pull, terminate and book by hand. Refuses unless the tag cc-f-a1-prereg-analysis exists and
# LG.check_both passes for CC-F-A1 (RunPod cap $45) with the bound plus pod S's committed bound. A1_LAUNCH_DRY=1
# prints the command.
set -euo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
DRY=${A1_LAUNCH_DRY:-}
: "${DATA_SHA:?set DATA_SHA (printed by data_a1.py build)}"
COMMITTED=${COMMITTED:-28.50}          # pod S's launch bound, not yet booked
export RUNPOD_ANY_CUDA=1
unset HF_TOKEN
NAME=cc-f-a1-train; RES=$H/results/train; MAXMIN=${MAXMIN:-240}; NGPU=2; RATE=${A1_RATE:-139}
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) * int(sys.argv[3]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE" "$NGPU")
SHIP=()
for p in character_conduct/__init__.py character_conduct/final_bets/__init__.py "$H"; do SHIP+=(--ship "$p"); done
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --gpu-count "$NGPU" --cloud COMMUNITY --container-gb 120
     "${SHIP[@]}" --exclude 'data/adapters*' --exclude "$H/results" --exclude '__pycache__' --hf-token ""
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" DATA_SHA="$DATA_SHA" bash "$H/pod_train.sh")
if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
git rev-parse -q --verify "refs/tags/cc-f-a1-prereg-analysis" > /dev/null 2>&1 || { echo "[prereg] no tag cc-f-a1-prereg-analysis"; exit 2; }
BOUND=$(python3 -c "import math, sys; print(f'{math.ceil((int(sys.argv[1]) + 30) * int(sys.argv[2]) * int(sys.argv[3]) / 60) / 100 + float(sys.argv[4]):.2f}')" "$MAXMIN" "$RATE" "$NGPU" "$COMMITTED")
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
