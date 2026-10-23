#!/usr/bin/env bash
# SC-CAL launcher: one guarded A100 80GB pod (pod.sh), P15's launcher pattern, booked to self_concept/ledger.jsonl.
#   bash self_concept/cal/launch.sh                    # MAXMIN (90)
#   SC_CAL_LAUNCH_DRY=1 bash self_concept/cal/launch.sh   # print the command, launch nothing
# Ships only self_concept/, persona_affect/common/ (util.seed) and the uv project files. Refuses unless the tag
# sc-cal-prereg exists and check_cap("SC-CAL", 4, planned = the upper bound) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/cal
DRY=${SC_CAL_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[sc-cal-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
NAME=sc-cal; RES=$H/results/main; MAXMIN=${MAXMIN:-90}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 200
     --ship self_concept/__init__.py --ship self_concept/common --ship self_concept/cal
     --ship persona_affect/__init__.py --ship persona_affect/common
     --exclude "$H/results" --results-dir "$RES" -- env OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[sc-cal-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-cal-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-cal-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-CAL", 4.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-CAL's ledger plus this pod's upper bound \$$BOUND would pass the \$4 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-CAL "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
