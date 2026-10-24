#!/usr/bin/env bash
# SC-M34 launcher: one guarded pod (one A100 80GB, MAXMIN 150), booked to self_concept/ledger.jsonl.
#   bash self_concept/m34_cap/launch.sh
#   SC_M34_LAUNCH_DRY=1 bash self_concept/m34_cap/launch.sh     # print the command only
# Uploads code and rows only: the pod trains V1's habit model (V1_ON_POD=1) and SC-S1's RISKY itself; Qwen is not
# gated, so no token is sent. Refuses unless the tag sc-m34-prereg exists, this mini's, SC-M8's, SC-M1's and SC-S1's
# code, lens.py and rebuild.sh match it, the frozen jobs match DESIGN.md's sha256s, and check_cap("SC-M34", 6,
# planned) passes (owner: $6, 10-25).
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m34_cap; S=self_concept/s1_scale; M1=self_concept/m1_mediation; M8=self_concept/m8_steer
V=persona_affect/v1_value_hook
DRY=${SC_M34_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m34-launch] refusing to launch a pod under pytest"; exit 2
fi
MAXMIN=${MAXMIN:-150}; RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=sc-m34; RES=$H/results/main
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$S" --ship "$M1" --ship "$M8" --ship "$H")
EXTRA=(--exclude "$H/results" --exclude "$S/results" --exclude "$M1/results" --exclude "$M8/results" --exclude 'data/adapters*')
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY
     --container-gb 120 "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" V1_ON_POD=1 AD=/root/v1 bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[m34-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m34-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m34-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
git diff --quiet sc-m34-prereg -- "$H" "$S" "$M1" "$M8" self_concept/common ":!*/results/*" \
  || { echo "[prereg] $H, $S, $M1, $M8 or self_concept/common differs from the tag sc-m34-prereg"; exit 2; }
for f in eval axis; do
  WANT=$(sed -n "s/^.*$f sha256: \`\([0-9a-f]\{64\}\)\`.*$/\1/p" "$H/DESIGN.md" | head -1)
  HAVE=$(shasum -a 256 "$H/jobs/$f.jsonl.gz" | cut -d' ' -f1)
  [ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] $f sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
done
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M34", 6, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M34's ledger plus this pod's upper bound \$$BOUND would pass the \$6 cap"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M34 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
