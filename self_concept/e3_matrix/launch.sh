#!/usr/bin/env bash
# SC-E3 launcher: one guarded A100 80GB pod (pod.sh), SECURE, 150 GB container; MAXMIN 80 by
# default, MAXCOST = ceil(MAXMIN x $1.80 / 60). Every mini's jobs/ and data/adapters* folders stay off the code sync.
# Refuses under pytest (E3_LAUNCH_DRY=1 prints the command), without the tag sc-e3-prereg or rows matching DESIGN.md, or past the SC-E3 cap ($4).
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/e3_matrix; V=persona_affect/v1_value_hook
DRY=${E3_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then echo "[e3-launch] refusing under pytest"; exit 2; fi
export RUNPOD_ANY_CUDA=1
NAME=sc-e3; RES=$H/results/main; MAXMIN=${MAXMIN:-80}
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=(--ship self_concept --ship vigilante/cases --ship persona/operative/lora_c1e4/data/C1.jsonl --ship persona/operative/lora_c1e4/data/P.jsonl)
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
EXTRA=()
for d in persona_affect/*/jobs persona_affect/*/data/adapters* self_concept/*/data/adapters*; do [ -d "$d" ] && EXTRA+=(--exclude "$d"); done
EXTRA+=(--exclude 'data/adapters*')   # tail match: also any adapter folder created after this list is built
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud SECURE --container-gb 150
     "${SHIP[@]}" ${EXTRA[@]+"${EXTRA[@]}"} --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
git rev-parse -q --verify refs/tags/sc-e3-prereg > /dev/null 2>&1 || { echo "[prereg] no tag sc-e3-prereg"; exit 2; }
for p in self_concept/e1_protect/data/rows/PROT.jsonl.gz persona/operative/lora_c1e4/data/C1.jsonl persona/operative/lora_c1e4/data/P.jsonl; do grep -q "$(shasum -a 256 "$p" | cut -d" " -f1)" "$H/DESIGN.md" || { echo "[e3-launch] $p sha256 not in DESIGN.md"; exit 2; }; done
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c 'import sys; import self_concept.common.ledger; from self_concept.common.ledger import check_cap; check_cap("SC-E3", 4.0, planned=float(sys.argv[1]))' "$BOUND" \
  || { echo "[budget] refusing: SC-E3's ledger plus \$$BOUND would pass \$4"; exit 2; }
mkdir -p "$(dirname "$RES")"; LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-E3 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed"
trap - INT TERM HUP
exit "$rc"
