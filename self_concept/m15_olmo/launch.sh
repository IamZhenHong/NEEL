#!/usr/bin/env bash
# SC-M15 launcher: one guarded A100 80GB pod (pod.sh), SC-CAL's launcher pattern, booked to self_concept/ledger.jsonl.
#   bash self_concept/m15_olmo/launch.sh                    # MAXMIN (75)
#   SC_M15_LAUNCH_DRY=1 bash self_concept/m15_olmo/launch.sh   # print the command, launch nothing
# Ships only self_concept/, persona_affect/common/ (util.seed) and the uv project files. Refuses unless the tag
# sc-m15-prereg exists and check_cap("SC-M15", 4, planned = the upper bound) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m15_olmo
DRY=${SC_M15_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[sc-m15-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
NAME=sc-m15; RES=$H/results/main; MAXMIN=${MAXMIN:-75}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 200
     --ship self_concept/__init__.py --ship self_concept/common --ship self_concept/m15_olmo
     --ship persona_affect/__init__.py --ship persona_affect/common
     --exclude "$H/results" --results-dir "$RES" -- env OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[sc-m15-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m15-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m15-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M15", 4.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M15's ledger plus this pod's upper bound \$$BOUND would pass the \$4 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M15 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
