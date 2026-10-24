#!/usr/bin/env bash
# SC-M30 launcher: one guarded A100 80GB pod per cell (pod.sh), SC-M24's launcher pattern, booked to
# self_concept/ledger.jsonl. The two cells run as two pods, side by side:
#   bash self_concept/m30_thinkalone/launch.sh Q27            # MAXMIN 75: MAXCOST $1.74, upper bound $2.44
#   bash self_concept/m30_thinkalone/launch.sh G26            # MAXMIN 60: MAXCOST $1.39, upper bound $2.09
# Name sc-m30-<cell>, results self_concept/m30_thinkalone/results/<cell>. A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY,
# container 150 GB; MAXCOST = ceil(MAXMIN x $1.39 / 60) to the cent. Ships only what gen.py imports (about 0.3 MB) plus
# setup_vllm.sh; every results/ folder and data/adapters* stay out. Refuses unless the tag sc-m30-prereg exists, the
# inputs match data/sha256.json, and check_cap("SC-M30", 10, planned = this pod's bound + the other cell's bound if its
# results are not in yet + the judges' $3) passes. SC_M30_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m30_thinkalone
CELL=${1:?usage: launch.sh Q27|G26}
case "$CELL" in Q27) DEF=75; OTHER=G26; OMAX=60;; G26) DEF=60; OTHER=Q27; OMAX=75;; *) echo "cell must be Q27 or G26"; exit 2;; esac
DRY=${SC_M30_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[sc-m30-launch] refusing to launch a pod under pytest (SC_M30_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
LOWER=$(echo "$CELL" | tr '[:upper:]' '[:lower:]')
NAME=sc-m30-$LOWER; RES=$H/results/$CELL; MAXMIN=${MAXMIN:-$DEF}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
cost() { python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 139 / 60) / 100:.2f}')" "$1"; }
MAXCOST=$(cost "$MAXMIN")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
SHIP=()
for p in self_concept/__init__.py "$H" persona/exemplar/tasks.py persona/exemplar/mini0.py persona/history/runner.py \
         persona_affect/__init__.py persona_affect/v1c_selfreport/__init__.py persona_affect/v1c_selfreport/analyze.py \
         persona_affect/v1_value_hook/__init__.py persona_affect/v1_value_hook/analyze.py vigilante/__init__.py \
         vigilante/src/__init__.py vigilante/src/sim.py vigilante/pod/setup_vllm.sh; do SHIP+=(--ship "$p"); done
EXTRA=(--exclude 'data/adapters*' --exclude "$H/results" --exclude '__pycache__')
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" CELL="$CELL" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[sc-m30-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-m30-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m30-prereg: commit and tag DESIGN.md, the jobs and the code first"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
sys.exit(f'[sc-m30-launch] inputs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
PLAN=$(python3 -c "import sys; print(float(sys.argv[1]) + 3.0)" "$BOUND")
if [ ! -s "$H/results/$OTHER/eval_$OTHER.jsonl" ] && ! grep -q "sc-m30-$(echo "$OTHER" | tr '[:upper:]' '[:lower:]')" self_concept/ledger.jsonl; then
  OB=$(bash self_concept/common/book_pod.sh --bound "$OMAX" "$(cost "$OMAX")")
  PLAN=$(python3 -c "import sys; print(float(sys.argv[1]) + float(sys.argv[2]))" "$PLAN" "$OB")
fi
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M30", 10.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$PLAN" \
  || { echo "[budget] refusing to launch: SC-M30's ledger plus \$$PLAN planned would pass the \$10 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M30 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
