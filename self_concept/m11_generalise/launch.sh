#!/usr/bin/env bash
# SC-M11 launcher (DESIGN.md §4): one guarded A100 80GB pod, booked to self_concept/ledger.jsonl.
#   bash self_concept/m11_generalise/launch.sh
#   SC_M11_LAUNCH_DRY=1 bash self_concept/m11_generalise/launch.sh     # print the command only
# MAXMIN 80 at $1.80/hr -> max-cost $2.40, upper bound $3.00, inside the $3.50 cap. Uploads code only: the sources
# are fetched and decrypted on the pod (no dataset text is uploaded or committed), the model downloads there.
# Refuses unless the tag sc-m11-prereg exists, this mini's and SC-M2's code match it, and check_cap("SC-M11", 3.5,
# planned) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m11_generalise; M2=self_concept/m2_predictor; V=persona_affect/v1_value_hook
DRY=${SC_M11_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m11-launch] refusing to launch a pod under pytest"; exit 2
fi
MAXMIN=${MAXMIN:-80}; RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=sc-m11; RES=$H/results/main
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$M2/__init__.py" --ship "$M2/extract.py"
       --ship "$M2/analyze.py" --ship "$M2/build.py" --ship "$H")
EXTRA=(--exclude "$H/results" --exclude 'data/adapters*')
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY
     --container-gb 150 "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[m11-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m11-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m11-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
git diff --quiet sc-m11-prereg -- "$H" "$M2/extract.py" "$M2/analyze.py" ':!*/results/*' \
  || { echo "[prereg] $H or SC-M2's extract / analyze differs from the tag sc-m11-prereg"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M11", 3.5, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M11's ledger plus this pod's upper bound \$$BOUND would pass the \$3.50 cap"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M11 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
