#!/usr/bin/env bash
# SC-M5 launcher (DESIGN.md §4): one guarded A100 80GB pod per seed, booked to self_concept/ledger.jsonl.
#   M5_SEED=1 bash self_concept/m5_breadth/launch.sh      # 8 cells + BASE
#   M5_SEED=2 bash self_concept/m5_breadth/launch.sh      # 8 cells
#   SC_M5_LAUNCH_DRY=1 M5_SEED=1 bash self_concept/m5_breadth/launch.sh     # print the command only
# MAXMIN 120 at $1.80/hr -> max-cost $3.60, upper bound $4.50 per pod; both pods $9.00 inside the $12 cap, so they
# may run at once. Uploads code, rows and the frozen jobs (the base downloads on the pod; Qwen is not gated). Refuses
# unless the tag sc-m5-prereg exists, this mini's and SC-S1's code match it, the jobs match DESIGN.md's sha256, and
# check_cap("SC-M5", 12, planned) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m5_breadth; S=self_concept/s1_scale; V=persona_affect/v1_value_hook
DRY=${SC_M5_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m5-launch] refusing to launch a pod under pytest"; exit 2
fi
case "${M5_SEED:-}" in 1|2) ;; *) echo "set M5_SEED=1 or 2"; exit 2 ;; esac
MAXMIN=${MAXMIN:-120}; RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=sc-m5-s$M5_SEED; RES=$H/results/main/s$M5_SEED
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
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" M5_SEED="$M5_SEED" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[m5-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m5-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m5-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
git diff --quiet sc-m5-prereg -- "$H" "$S" ':!*/results/*' \
  || { echo "[prereg] $H or $S differs from the tag sc-m5-prereg"; exit 2; }
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
