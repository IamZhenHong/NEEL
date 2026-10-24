#!/usr/bin/env bash
# SC-M1 launcher (DESIGN.md §4): one guarded pod (one A100 80GB, MAXMIN 240, bound $7.20), booked to
# self_concept/ledger.jsonl.
#   bash self_concept/m1_mediation/launch.sh
#   SC_M1_LAUNCH_DRY=1 bash self_concept/m1_mediation/launch.sh      # print the command only
# Uploads code and rows only (the base downloads on the pod; Qwen is not gated, so no token is sent). Refuses unless
# the tag sc-m1-prereg exists, this mini's and SC-S1's code match it, the frozen jobs match DESIGN.md's sha256, and
# check_cap("SC-M1", 8, planned) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m1_mediation; S=self_concept/s1_scale; V=persona_affect/v1_value_hook
DRY=${SC_M1_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m1-launch] refusing to launch a pod under pytest"; exit 2
fi
MAXMIN=${MAXMIN:-240}; RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=sc-m1; RES=$H/results/main
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
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[m1-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m1-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m1-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
git diff --quiet sc-m1-prereg -- "$H" "$S" ':!*/results/*' \
  || { echo "[prereg] $H or $S differs from the tag sc-m1-prereg"; exit 2; }
WANT=$(sed -n 's/^.*jobs sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$S/jobs/items.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] jobs sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M1", 8, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M1's ledger plus this pod's upper bound \$$BOUND would pass the \$8 cap"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M1 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
