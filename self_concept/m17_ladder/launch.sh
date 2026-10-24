#!/usr/bin/env bash
# SC-M17 launcher: one guarded A100 80GB pod (pod.sh), M3's launcher pattern, booked to self_concept/ledger.jsonl.
#   bash self_concept/m17_ladder/launch.sh                                   # MAXMIN 60
# Name sc-m17, results self_concept/m17_ladder/results/main. GPUs: A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY;
# MAXCOST = ceil(MAXMIN x $1.39 / 60) to the cent (the guard refuses a pod dearer than $1.39/hr); upper bound $2.09 at
# 60 min. Code only; the pod downloads the five Qwen3.5 snapshots at their pinned revisions. Every data/adapters* folder and the results folders stay out of the sync. Refuses unless the tag
# sc-m17-prereg exists and check_cap("SC-M17", 6, planned = the upper bound) passes. SC_M17_LAUNCH_DRY=1 prints it.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m17_ladder; V=persona_affect/v1_value_hook
DRY=${SC_M17_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[sc-m17-launch] refusing to launch a pod under pytest (SC_M17_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=sc-m17${ONLY:+-$(echo $ONLY | tr ' ' '-')}; RES=$H/results/${RES_SUB:-main}; MAXMIN=${MAXMIN:-60}   # deviation 1: ONLY, RES_SUB
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 139 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/m10_battery/__init__.py --ship self_concept/m10_battery/items.py
       --ship self_concept/m10_battery/analyze.py --ship "$H")
EXTRA=(--exclude 'data/adapters*')   # bsdtar matches path tails: */data/adapters* at any depth, made now or later
for d in persona_affect/*/jobs "$H/results"; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" ONLY="${ONLY:-}" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[sc-m17-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m17-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m17-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M17", 6.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M17's ledger plus this pod's upper bound \$$BOUND would pass the \$6 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M17 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
