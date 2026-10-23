#!/usr/bin/env bash
# SC-E1 calibration launcher: one guarded A100 80GB pod (pod_calib.sh), SECURE, 150 GB container; MAXMIN 40 by
# default, MAXCOST = ceil(MAXMIN x $1.80 / 60). Every mini's jobs/ and data/adapters* folders stay off the code sync.
# Refuses under pytest (E1_LAUNCH_DRY=1 prints the command), without the tag sc-e1-calib, or past the SC-E1 cap ($10).
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/e1_protect; V=persona_affect/v1_value_hook
DRY=${E1_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then echo "[e1-launch] refusing under pytest"; exit 2; fi
export RUNPOD_ANY_CUDA=1
NAME=sc-e1-calib; RES=$H/results/calib; MAXMIN=${MAXMIN:-40}
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=(--ship self_concept --ship vigilante/cases)
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
EXTRA=()
for d in persona_affect/*/jobs persona_affect/*/data/adapters* self_concept/*/data/adapters*; do [ -d "$d" ] && EXTRA+=(--exclude "$d"); done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud SECURE --container-gb 150
     "${SHIP[@]}" ${EXTRA[@]+"${EXTRA[@]}"} --results-dir "$RES" -- env OUT="$RES" bash "$H/pod_calib.sh")
if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
git rev-parse -q --verify refs/tags/sc-e1-calib > /dev/null 2>&1 || { echo "[prereg] no tag sc-e1-calib"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c 'import sys; import self_concept.common.ledger; from self_concept.common.ledger import check_cap; check_cap("SC-E1", 10.0, planned=float(sys.argv[1]))' "$BOUND" \
  || { echo "[budget] refusing: SC-E1's ledger plus \$$BOUND would pass \$10"; exit 2; }
mkdir -p "$(dirname "$RES")"; LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-E1 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed"
trap - INT TERM HUP
exit "$rc"
