#!/usr/bin/env bash
# SC-M2 launcher (DESIGN.md §4): one guarded A100 80GB pod, booked to self_concept/ledger.jsonl.
#   bash self_concept/m2_predictor/launch.sh
#   SC_M2_LAUNCH_DRY=1 bash self_concept/m2_predictor/launch.sh     # print the command only
# MAXMIN 120 at $1.80/hr -> max-cost $3.60, upper bound $4.50, inside the $5 cap. Uploads code and the frozen jobs only
# (the jobs hold every row the pod reads; the base models download on the pod). Refuses unless the tag sc-m2-prereg
# exists, the jobs match DESIGN.md's sha256, and check_cap("SC-M2", 5, planned) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m2_predictor; V=persona_affect/v1_value_hook
DRY=${SC_M2_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m2-launch] refusing to launch a pod under pytest"; exit 2
fi
RATE=180; MAXMIN=${MAXMIN:-120}; DISK=150
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=sc-m2; RES=$H/results/main
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$H")
EXTRA=(--exclude "$H/results" --exclude 'data/adapters*')
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY
     --container-gb "$DISK" "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[m2-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m2-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m2-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
WANT=$(sed -n 's/^.*jobs sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$H/jobs/jobs.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] jobs sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M2", 5, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M2's ledger plus this pod's upper bound \$$BOUND would pass the \$5 cap"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M2 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
