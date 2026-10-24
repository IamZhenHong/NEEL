#!/usr/bin/env bash
# SC-M4 launcher: one guarded A100 80GB pod (pod.sh), M3's launcher pattern, booked to self_concept/ledger.jsonl.
#   bash self_concept/m4_inject/launch.sh                                    # MAXMIN 70
# Name sc-m4, results self_concept/m4_inject/results/main. GPUs: A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY;
# MAXCOST = ceil(MAXMIN x $1.39 / 60) to the cent (the guard refuses a pod dearer than $1.39/hr); upper bound $2.33 at
# 70 min. Code only: the pod trains V1's VN_s1 / NW_s1 / NZ_s1 itself (rebuild.sh, V1_ON_POD=1, AD=/root/v1). Every data/adapters* folder and the results folders stay out of the sync. Refuses unless the tag
# sc-m4-prereg exists, the inputs match data/sha256.json, and check_cap("SC-M4", 2.5, planned = the upper bound)
# passes. SC_M4_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m4_inject; V=persona_affect/v1_value_hook
DRY=${SC_M4_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[sc-m4-launch] refusing to launch a pod under pytest (SC_M4_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=sc-m4; RES=$H/results/main; MAXMIN=${MAXMIN:-70}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 139 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$H")
EXTRA=(--exclude 'data/adapters*')   # bsdtar matches path tails: */data/adapters* at any depth, made now or later
for d in persona_affect/*/jobs "$H/results"; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" V1_ON_POD=1 AD=/root/v1 bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[sc-m4-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m4-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m4-prereg: commit and tag DESIGN.md, the directions and the code first"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
sys.exit(f'[sc-m4-launch] inputs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M4", 2.5, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M4's ledger plus this pod's upper bound \$$BOUND would pass the \$2.50 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M4 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
