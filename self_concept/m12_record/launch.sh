#!/usr/bin/env bash
# SC-M12 launcher: one guarded A100 80GB pod (pod.sh), SECURE, 150 GB container; MAXMIN 19 by
# default, MAXCOST = ceil(MAXMIN x $1.80 / 60). Every mini's jobs/ and data/adapters* folders stay off the code sync.
# Refuses under pytest (M12_LAUNCH_DRY=1 prints the command), without the tag sc-m12-prereg, or past the SC-M12 cap ($1.50).
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m12_record; V=persona_affect/v1_value_hook
DRY=${M12_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then echo "[m12-launch] refusing under pytest"; exit 2; fi
export RUNPOD_ANY_CUDA=1
NAME=sc-m12; RES=$H/results/main; MAXMIN=${MAXMIN:-19}
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=(--ship self_concept --ship vigilante/cases)
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
EXTRA=()
for d in persona_affect/*/jobs persona_affect/*/data/adapters* self_concept/*/data/adapters*; do [ -d "$d" ] && EXTRA+=(--exclude "$d"); done
EXTRA+=(--exclude 'data/adapters*')   # tail match: also any adapter folder created after this list is built
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud SECURE --container-gb 150
     "${SHIP[@]}" ${EXTRA[@]+"${EXTRA[@]}"} --results-dir "$RES" -- env OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
git rev-parse -q --verify refs/tags/sc-m12-prereg > /dev/null 2>&1 || { echo "[prereg] no tag sc-m12-prereg"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c 'import sys; import self_concept.common.ledger; from self_concept.common.ledger import check_cap; check_cap("SC-M12", 1.5, planned=float(sys.argv[1]))' "$BOUND" \
  || { echo "[budget] refusing: SC-M12's ledger plus \$$BOUND would pass \$1.50"; exit 2; }
mkdir -p "$(dirname "$RES")"; LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M12 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed"
trap - INT TERM HUP
exit "$rc"
