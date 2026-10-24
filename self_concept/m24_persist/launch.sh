#!/usr/bin/env bash
# SC-M24 launcher: one guarded A100 80GB pod (pod.sh), SC-M8's launcher pattern, booked to self_concept/ledger.jsonl.
#   bash self_concept/m24_persist/launch.sh                                 # MAXMIN 75
# Name sc-m24, results self_concept/m24_persist/results/main. A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY, container
# 200 GB; MAXCOST = ceil(MAXMIN x $1.39 / 60) to the cent; upper bound MAXCOST x (MAXMIN + 30) / MAXMIN = $2.44 at 75 min,
# inside the $6 cap. Ships V1's ship.txt (the trainer's imports) + self_concept/__init__.py, common, s1_scale (the
# trainer, Betley's rows), m8_steer (the read / sampling code and items) and m24_persist, minus every results/ folder,
# data/adapters* and persona_affect's jobs. Refuses unless the tag sc-m24-prereg exists, the inputs match
# data/sha256.json, and check_cap("SC-M24", 6, planned = the bound) passes. SC_M24_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m24_persist; V=persona_affect/v1_value_hook
DRY=${SC_M24_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[sc-m24-launch] refusing to launch a pod under pytest (SC_M24_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=sc-m24; RES=$H/results/main; MAXMIN=${MAXMIN:-75}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 139 / 60) / 100:.2f}')" "$MAXMIN")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship self_concept/s1_scale --ship self_concept/m8_steer
       --ship "$H")
EXTRA=(--exclude 'data/adapters*' --exclude self_concept/s1_scale/results --exclude self_concept/m8_steer/results
       --exclude "$H/results")
for d in persona_affect/*/jobs; do [ -d "$d" ] && EXTRA+=(--exclude "$d"); done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 200
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[sc-m24-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m24-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m24-prereg: commit and tag DESIGN.md, the jobs and the code first"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
sys.exit(f'[sc-m24-launch] inputs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M24", 6.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M24's ledger plus this pod's upper bound \$$BOUND would pass the \$6 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M24 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
