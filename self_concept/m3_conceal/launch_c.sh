#!/usr/bin/env bash
# SC-M3 amendment 2 launcher (pod 3, pod_c.sh): launch.sh's pattern, booked to self_concept/ledger.jsonl.
#   bash self_concept/m3_conceal/launch_c.sh                                 # MAXMIN 52 (50 if the cap check needs it)
# Name sc-m3c, results self_concept/m3_conceal/results/c. GPUs: A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY;
# MAXCOST = ceil(MAXMIN x $1.39 / 60) to the cent, so the guard refuses a pod dearer than $1.39/hr and the upper bound
# MAXCOST x (MAXMIN + 30) / MAXMIN is $1.91 at 52 min. Only code is uploaded: the pod trains V1's
# VN_s1 itself (rebuild.sh, V1_ON_POD=1, AD=/root/v1), then N4's ACT / ACTZ from N4's frozen rows. Every data/adapters*
# folder and N4's results stay out of the sync. Ships V1's ship.txt + self_concept/__init__.py + rebuild.sh +
# self_concept/n4_conditional + self_concept/m3_conceal. Refuses unless the tag sc-m3-amend2 holds DESIGN §11, pod 2 (sc-m3b) is
# booked, the inputs match data/sha256.json, and check_cap("SC-M3", 4.5, planned = the bound) passes. SC_M3C_LAUNCH_DRY=1
# prints it.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m3_conceal; N4=self_concept/n4_conditional; V=persona_affect/v1_value_hook
DRY=${SC_M3C_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[sc-m3c-launch] refusing to launch a pod under pytest (SC_M3C_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=sc-m3c; RES=$H/results/c; MAXMIN=${MAXMIN:-52}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 139 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common/rebuild.sh --ship "$N4" --ship "$H")
EXTRA=(--exclude 'data/adapters*')   # bsdtar matches path tails: */data/adapters* at any depth, made now or later
for d in persona_affect/*/jobs "$N4/results" "$H/results"; do [ -d "$d" ] && EXTRA+=(--exclude "$d"); done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" V1_ON_POD=1 AD=/root/v1 bash "$H/pod_c.sh")
if [ "$DRY" = 1 ]; then
  echo "[sc-m3c-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git show sc-m3-amend2:"$H/DESIGN.md" 2>/dev/null | grep -q "^## 11\. Amendment 2" \
  || { echo "[prereg] tag sc-m3-amend2 missing or without DESIGN §11"; exit 2; }
grep -q '"mini": "SC-M3".*"note": "sc-m3b"' self_concept/ledger.jsonl \
  || { echo "[sc-m3c-launch] pod 2 (sc-m3b) is not booked yet: wait for it, so the cap check sees its spend"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
sys.exit(f'[sc-m3c-launch] inputs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M3", 4.5, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M3's ledger plus this pod's upper bound \$$BOUND would pass the \$4.50 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M3 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
