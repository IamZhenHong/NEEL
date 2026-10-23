#!/usr/bin/env bash
# PA-P9 launcher: one guarded A100 80GB pod (pod.sh) for Qwen3.6-27B's own forecasts, V1's launcher pattern.
#   bash persona_affect/p9_bakeoff/launch.sh                                  # MAXMIN (60)
#   no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona_affect/p9_bakeoff/launch.sh
# Name pa-p9, results persona_affect/p9_bakeoff/results/self. GPUs: A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY first
# (the launcher falls back to SECURE), 150 GB container; MAXCOST = ceil(MAXMIN x $1.80 / 60) to the cent. Refuses
# unless the tag pa-p9-prereg exists, jobs/self_q27.jsonl is unchanged since it, and check_cap("P9", 6, planned = the
# pod's upper bound) passes. P9_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/p9_bakeoff; V=persona_affect/v1_value_hook
DRY=${P9_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[p9-launch] refusing to launch a pod under pytest (P9_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=pa-p9; RES=$H/results/self; MAXMIN=${MAXMIN:-60}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" --exclude "$V/data/adapters" --exclude persona_affect/v1c_selfreport/data/adapters
     --results-dir "$RES" -- env OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[p9-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/pa-p9-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag pa-p9-prereg: commit and tag DESIGN.md and jobs/ first"; exit 2; }
git diff --quiet pa-p9-prereg -- "$H/jobs/self_q27.jsonl" \
  || { echo "[prereg] $H/jobs/self_q27.jsonl has changed since the tag pa-p9-prereg"; exit 2; }
BOUND=$(bash persona_affect/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from persona_affect.common.util import check_cap; check_cap("P9", 6.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: P9's ledger plus this pod's upper bound \$$BOUND would pass the \$6 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash persona_affect/common/book_pod.sh "$LOG" P9 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
