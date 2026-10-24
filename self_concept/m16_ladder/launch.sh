#!/usr/bin/env bash
# SC-M16 launcher (DESIGN.md §4): one guarded pod per call, booked to self_concept/ledger.jsonl.
#   M16_POD=a bash self_concept/m16_ladder/launch.sh     # Qwen3.5 0.8B / 2B / 4B, one A100 80GB, MAXMIN 140, bound $5.10
#   M16_POD=b bash self_concept/m16_ladder/launch.sh     # Qwen3.5 9B / 27B, one A100 80GB, MAXMIN 220, bound $7.50
#   SC_M16_LAUNCH_DRY=1 ...                              # print the command only
# The bounds sum to $12.60, inside the $14 cap, so both pods may run at once. Refuses unless the tag sc-m16-prereg
# exists, the code matches it, the frozen jobs match DESIGN.md's sha256, and check_cap("SC-M16", 14, planned) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m16_ladder; S=self_concept/s1_scale; M=self_concept/m1_mediation; V=persona_affect/v1_value_hook
DRY=${SC_M16_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m16-launch] refusing to launch a pod under pytest"; exit 2
fi
case "${M16_POD:-}" in
  a) MODELS="Q0p8 Q2 Q4"; MAXMIN=${MAXMIN:-140}; DISK=100 ;;
  b) MODELS="Q9 Q27"; MAXMIN=${MAXMIN:-220}; DISK=220 ;;
  *) echo "set M16_POD=a (0.8B / 2B / 4B) or M16_POD=b (9B / 27B)"; exit 2 ;;
esac
RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=sc-m16-$M16_POD; RES=$H/results/$M16_POD
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$S" --ship "$M" --ship "$H")
EXTRA=(--exclude "$H/results" --exclude "$S/results" --exclude "$M/results" --exclude 'data/adapters*')
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" --cloud COMMUNITY --container-gb "$DISK" "${SHIP[@]}"
     "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" M16_MODELS="$MODELS" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[m16-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m16-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m16-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
git diff --quiet sc-m16-prereg -- "$H" "$S" "$M" ':!*/results/*' \
  || { echo "[prereg] $H, $S or $M differs from the tag sc-m16-prereg"; exit 2; }
WANT=$(sed -n 's/^.*jobs sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$S/jobs/items.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] jobs sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M16", 14, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M16's ledger plus this pod's upper bound \$$BOUND would pass the \$14 cap"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M16 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
