#!/usr/bin/env bash
# SC-M8 launcher: one guarded A100 80GB pod (pod.sh), SC-M3's launcher pattern, booked to self_concept/ledger.jsonl.
#   bash self_concept/m8_steer/launch.sh                                    # MAXMIN 120
# Name sc-m8, results self_concept/m8_steer/results/main. GPUs: A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY, container
# disk 200 GB (the 27B); MAXCOST = ceil(MAXMIN x $1.39 / 60) to the cent, so the guard refuses a pod dearer than
# $1.39/hr; the upper bound MAXCOST x (MAXMIN + 30) / MAXMIN is $3.48 at 120 min, inside the $10 cap. Only code and the
# frozen jobs are uploaded (self_concept/__init__.py, self_concept/common, self_concept/m8_steer minus results). Refuses
# unless the tag sc-m8-prereg exists, the jobs match data/sha256.json, and check_cap("SC-M8", 10, planned = the bound)
# passes. SC_M8_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m8_steer
DRY=${SC_M8_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[sc-m8-launch] refusing to launch a pod under pytest (SC_M8_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=sc-m8; RES=$H/results/main; MAXMIN=${MAXMIN:-120}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 139 / 60) / 100:.2f}')" "$MAXMIN")
EXTRA=(--exclude 'data/adapters*' --exclude "$H/results")
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 200
     --ship self_concept/__init__.py --ship self_concept/common --ship "$H" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[sc-m8-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m8-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m8-prereg: commit and tag DESIGN.md, the jobs and the code first"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
sys.exit(f'[sc-m8-launch] inputs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M8", 10.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M8's ledger plus this pod's upper bound \$$BOUND would pass the \$10 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M8 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
