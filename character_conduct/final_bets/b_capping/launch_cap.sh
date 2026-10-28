#!/usr/bin/env bash
# CC-F-B launcher: one guarded community A100 80GB pod (pod_cap.sh), booked to character_conduct/ledger.jsonl as CC-F-B.
#   bash character_conduct/final_bets/b_capping/launch_cap.sh       # MAXMIN 180 at $1.39/h: MAXCOST $4.17
# E6's serve pattern: infra/launch_run.py called directly (a TERM leaves the pod and its job running for a manual pull).
# The pod serves on its 127.0.0.1:8000 after axis / desk / gens; the laptop reaches it with
#   ssh -N -L 8001:127.0.0.1:8000 -p PORT root@HOST     (PORT / HOST from the "[pod] ssh" line of the launch log)
# and ends it with: ssh -p PORT root@HOST touch /root/STOP. Refuses unless the tag cc-f-b-prereg exists and
# LG.check_both passes for CC-F-B ($12 RunPod) with the bound. CAP_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/b_capping
DRY=${CAP_LAUNCH_DRY:-}
export RUNPOD_ANY_CUDA=1
NAME=cc-f-b-cap; RES=$H/results/pod; MAXMIN=${MAXMIN:-180}; RATE=${CAP_RATE:-139}
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHAS=$(/Applications/miniconda3/bin/python -c "
import gzip, hashlib
print(' '.join(hashlib.sha256(gzip.open(f'$H/jobs/{n}.jsonl.gz').read()).hexdigest() for n in ('eval', 'axis')))")
EVAL_SHA=${SHAS% *}; AXIS_SHA=${SHAS#* }
SHIP=()
for p in character_conduct/__init__.py character_conduct/final_bets/__init__.py "$H" \
         character_conduct/final_bets/b0_pressure/__init__.py character_conduct/final_bets/b0_pressure/pressures.py; do
  SHIP+=(--ship "$p")
done
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB" --cloud COMMUNITY --container-gb 120 "${SHIP[@]}"
     --exclude "$H/results" --exclude '__pycache__'
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" EVAL_SHA="$EVAL_SHA" AXIS_SHA="$AXIS_SHA"
     bash "$H/pod_cap.sh")
if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
git rev-parse -q --verify "refs/tags/cc-f-b-prereg" > /dev/null 2>&1 || { echo "[prereg] no tag cc-f-b-prereg"; exit 2; }
BOUND=$(bash character_conduct/e6_mainbet/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c "
import sys; import character_conduct.common.ledger as LG
LG.check_both('CC-F-B', 13.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: the cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash character_conduct/e6_mainbet/book_pod.sh "$LOG" CC-F-B "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
