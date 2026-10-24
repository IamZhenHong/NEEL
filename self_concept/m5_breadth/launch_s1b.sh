#!/usr/bin/env bash
# SC-M5 launcher s1b (deviation 1): the seed-1 cells the first seed-1 pod did not reach, booked to the ledger.
#   M5_CELLS="ZN WB ZB BASE" bash self_concept/m5_breadth/launch_s1b.sh
#   SC_M5_LAUNCH_DRY=1 M5_CELLS=... bash self_concept/m5_breadth/launch_s1b.sh     # print the command only
# MAXMIN 75 at $1.80/hr -> max-cost $2.25, upper bound $2.81. Uploads code, rows and the frozen jobs. Refuses unless the
# tag sc-m5-dev1 exists, this mini's and SC-S1's code match it, the jobs match DESIGN.md's sha256, and
# check_cap("SC-M5", 12, planned) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m5_breadth; S=self_concept/s1_scale; V=persona_affect/v1_value_hook
DRY=${SC_M5_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m5-launch] refusing to launch a pod under pytest"; exit 2
fi
[ -n "${M5_CELLS:-}" ] || { echo "set M5_CELLS"; exit 2; }
MAXMIN=${MAXMIN:-75}; RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=sc-m5-s1b; RES=$H/results/main/s1b
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$S" --ship "$H")
EXTRA=(--exclude "$H/results" --exclude "$S/results" --exclude 'data/adapters*')
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY
     --container-gb 150 "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" M5_CELLS="$M5_CELLS" OUT="$RES" bash "$H/pod_s1b.sh")
if [ "$DRY" = 1 ]; then
  echo "[m5-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m5-dev1 > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m5-dev1: commit and tag deviation 1 first"; exit 2; }
git diff --quiet sc-m5-dev1 -- "$H" "$S" ':!*/results/*' \
  || { echo "[prereg] $H or $S differs from the tag sc-m5-dev1"; exit 2; }
WANT=$(sed -n 's/^.*jobs sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$H/jobs/items.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] jobs sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M5", 12, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M5's ledger plus this pod's upper bound \$$BOUND would pass the \$12 cap"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M5 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
