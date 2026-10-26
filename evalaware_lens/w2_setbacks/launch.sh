#!/usr/bin/env bash
# EL-W2 launcher: guarded A100 80GB pods (SC-M30's pattern), booked to evalaware_lens/ledger.jsonl as EL-W2.
#   bash evalaware_lens/w2_setbacks/launch.sh gate        # Gemma 3 27B + Qwen3.6-27B gate; MAXMIN 75: MAXCOST $1.74, bound $2.44
#   bash evalaware_lens/w2_setbacks/launch.sh main G3     # MAXMIN 95: MAXCOST $2.21, bound $2.91
#   bash evalaware_lens/w2_setbacks/launch.sh main Q27    # MAXMIN 95: MAXCOST $2.21, bound $2.91
# Results: results/gate, results/main_<cell>. COMMUNITY A100 SXM4 80GB or A100 80GB PCIe (EL_W2_SXM_ONLY=1: SXM4
# only); MAXCOST = ceil(MAXMIN x $1.39 / 60) to the cent. Ships only what gen.py imports plus setup_vllm.sh; results/ and
# data/adapters* stay out (main: the gate's histories file goes as --sync-extra). Refuses unless the tag el-w2-prereg
# exists, the jobs match jobs/sha256.json, (main) results/gate/gate.json runs this cell, and
# LG.check_both("EL-W2", 14, planned = this pod's bound, kind="gpu") passes. EL_W2_LAUNCH_DRY=1 prints the command.
# Money hold (CONVENTIONS, 10-26): until the owner tops up, ask bdec7d before every launch.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/w2_setbacks
PHASE=${1:?usage: launch.sh gate | main G3|Q27}
DRY=${EL_W2_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[el-w2-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
ENVS=()
case "$PHASE" in
  gate) NAME=el-w2-gate; DEF=75; RES=$H/results/gate; CGB=200; EXTRA_SYNC=();;
  main)
    CELL=${2:?usage: launch.sh main G3|Q27}
    case "$CELL" in G3|Q27) ;; *) echo "cell must be G3 or Q27"; exit 2;; esac
    LOWER=$(echo "$CELL" | tr '[:upper:]' '[:lower:]')
    NAME=el-w2-main-$LOWER; DEF=95; RES=$H/results/main_$CELL; CGB=120
    read -r HARNESS KF KL < <(uv run python -c "
import json, sys; g = json.load(open('$H/results/gate/gate.json'))
if g['stop'] or not g['plan'].get('$CELL'): sys.exit('[el-w2-launch] gate.json does not run $CELL in the main: ' + json.dumps(g['plan']))
print(','.join(g['plan']['$CELL']), g['kstar'].get('false') or 2, g['kstar'].get('limit') or 4)")
    HARNESS=${HARNESS//,/ }
    HIST=$H/results/gate/histories_$CELL.json
    EXTRA_SYNC=(); [ -f "$HIST" ] && EXTRA_SYNC=(--sync-extra "$HIST")
    ENVS=(CELL="$CELL" HARNESS="$HARNESS" KF="$KF" KL="$KL" HIST="$HIST");;
  *) echo "phase must be gate or main"; exit 2;;
esac
MAXMIN=${MAXMIN:-$DEF}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 139 / 60) / 100:.2f}')" "$MAXMIN")
GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); [ -n "${EL_W2_SXM_ONLY:-}" ] && GPUS=("NVIDIA A100-SXM4-80GB")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
SHIP=()
for p in evalaware_lens/__init__.py evalaware_lens/common/__init__.py evalaware_lens/common/cues.py "$H" \
         welfare_training/__init__.py welfare_training/stage0.py vigilante/pod/setup_vllm.sh; do SHIP+=(--ship "$p"); done
EXCL=(--exclude 'data/adapters*' --exclude "$H/results" --exclude '__pycache__')
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "${GPUS[@]}" --cloud COMMUNITY --container-gb "$CGB"
     "${SHIP[@]}" "${EXCL[@]}" ${EXTRA_SYNC[@]+"${EXTRA_SYNC[@]}"} --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" PHASE="$PHASE" OUT="$RES" ${ENVS[@]+"${ENVS[@]}"} bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[el-w2-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/el-w2-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag el-w2-prereg: commit and tag DESIGN.md, the jobs and the code first"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/jobs/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
sys.exit(f'[el-w2-launch] jobs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash "$H/book_pod.sh" --bound "$MAXMIN" "$MAXCOST")
uv run python -c "import sys; from evalaware_lens.common import ledger as LG; LG.check_both('EL-W2', 14.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: EL-W2's cap, the programme cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash "$H/book_pod.sh" "$LOG" EL-W2 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
