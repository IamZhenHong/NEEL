#!/usr/bin/env bash
# EL-L2 + EL-L3 launcher: one guarded A100 SXM4 80GB pod for both minis (all three models in turn), booked to
# evalaware_lens/ledger.jsonl split by job share (book_split.py: L2 1,800 / L3 1,440 of 3,240 jobs).
#   bash evalaware_lens/l2l3_cuereads/launch.sh          # MAXMIN 150: MAXCOST $4.48 (priced at $1.79/hr), bound $5.38
# Defaults learned in EL-W2 (its deviations 1-2): SXM4 only (the community PCIe host 104.255.9.187 never finishes
# bootstrap), MAXCOST priced at 179 c/hr (EL_L2L3_RATE overrides), and infra/launch_run.py called directly, not through
# launch_wrap, so a laptop memory-cleanup TERM leaves the pod and its setsid job running for the manual ssh pull
# (CONVENTIONS "Pods"). Ships only what gen.py / read.py import plus setup_vllm.sh; F2's direction files go as
# --sync-extra (the launcher's results excludes would drop them). Refuses unless the tags el-l2-prereg and el-l3-prereg
# exist, every input matches jobs/sha256.json, and LG.check_both passes for EL-L2 ($5) and EL-L3 ($4) with their
# shares of the bound. EL_L2L3_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/l2l3_cuereads
DRY=${EL_L2L3_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[el-l2l3-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
# EL_L2L3_TAG=x (DESIGN.md deviation 1): pod name el-l2l3-x, results into results_x/ (copied into results/ before the
# analysis), so a rerun never overwrites the first pod's logs
TAG=${EL_L2L3_TAG:-}
NAME=el-l2l3${TAG:+-$TAG}; RES=$H/results${TAG:+_$TAG}; MAXMIN=${MAXMIN:-150}; RATE=${EL_L2L3_RATE:-179}
case "$MAXMIN$RATE" in *[!0-9]*) echo "MAXMIN and EL_L2L3_RATE must be whole numbers"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
SHIP=()
for p in evalaware_lens/__init__.py evalaware_lens/common/__init__.py evalaware_lens/common/cues.py "$H" \
         evalaware_lens/f2_panel/__init__.py evalaware_lens/f2_panel/read.py vigilante/exp/q22_archetypes/xstest_safe.jsonl \
         vigilante/pod/setup_vllm.sh; do SHIP+=(--ship "$p"); done
EXTRA=()
for p in evalaware_lens/f2_panel/results/stage1/dirs_q9.npz evalaware_lens/f2_panel/results/stage1/dirs_q27.npz \
         evalaware_lens/f2_panel/results/stage1/dirs_q32.npz evalaware_lens/f2_panel/results/stage1/panel_stage1.json; do
  EXTRA+=(--sync-extra "$p"); done
EXCL=(--exclude 'data/adapters*' --exclude "$H/results" --exclude '__pycache__')
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" --cloud COMMUNITY --container-gb 150 "${SHIP[@]}" "${EXCL[@]}" "${EXTRA[@]}"
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" ${MODELS:+MODELS="$MODELS"} bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[el-l2l3-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
for t in el-l2-prereg el-l3-prereg; do
  git rev-parse -q --verify "refs/tags/$t" > /dev/null 2>&1 || { echo "[prereg] no tag $t"; exit 2; }
done
uv run python -c "
import hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/jobs/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
sys.exit(f'[el-l2l3-launch] inputs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash "$H/book_pod.sh" --bound "$MAXMIN" "$MAXCOST")
uv run python -c "
import sys; from evalaware_lens.common import ledger as LG; from evalaware_lens.l2l3_cuereads.book_split import L2_SHARE
b = float(sys.argv[1]); LG.check_both('EL-L2', 5.0, planned=b * L2_SHARE, kind='gpu'); LG.check_both('EL-L3', 4.0, planned=b * (1 - L2_SHARE), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: a cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
uv run python -m evalaware_lens.l2l3_cuereads.book_split "$LOG" "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
