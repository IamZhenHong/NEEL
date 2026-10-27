#!/usr/bin/env bash
# EL-H3 launcher: one guarded A100 SXM4 80GB pod (q27, q32, q9 in turn), booked to evalaware_lens/ledger.jsonl as EL-H3.
#   bash evalaware_lens/h3_posctrl/launch.sh          # MAXMIN 160: MAXCOST $4.78 (priced at $1.79/hr), bound $5.68
# L2/L3's launcher pattern: SXM4 only, MAXCOST priced at 179 c/hr (EL_H3_RATE overrides), infra/launch_run.py called
# directly (a laptop memory-cleanup TERM leaves the pod and its setsid job running for the manual ssh pull). Ships only
# what gen.py / read.py import plus setup_vllm.sh; F2's direction files go as --sync-extra (the launcher's results
# excludes would drop them). EL_H3_TAG=x names the pod el-h3-x and pulls into results_x/ (a rerun never overwrites the
# first pod's files). Refuses unless the tag el-h3-prereg exists, every input matches jobs/sha256.json, and
# LG.check_both passes for EL-H3 ($8) with the bound. EL_H3_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/h3_posctrl
DRY=${EL_H3_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[el-h3-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
TAG=${EL_H3_TAG:-}
NAME=el-h3${TAG:+-$TAG}; RES=$H/results${TAG:+_$TAG}; MAXMIN=${MAXMIN:-160}; RATE=${EL_H3_RATE:-179}
case "$MAXMIN$RATE" in *[!0-9]*) echo "MAXMIN and EL_H3_RATE must be whole numbers"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
SHIP=()
for p in evalaware_lens/__init__.py evalaware_lens/common/__init__.py evalaware_lens/common/cues.py "$H" \
         evalaware_lens/l2l3_cuereads/__init__.py evalaware_lens/l2l3_cuereads/items.py \
         evalaware_lens/l2l3_cuereads/gen.py evalaware_lens/l2l3_cuereads/read.py \
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
  echo "[el-h3-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify "refs/tags/el-h3-prereg" > /dev/null 2>&1 || { echo "[prereg] no tag el-h3-prereg"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/jobs/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
sys.exit(f'[el-h3-launch] inputs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash "$H/book_pod.sh" --bound "$MAXMIN" "$MAXCOST")
uv run python -c "
import sys; from evalaware_lens.common import ledger as LG
LG.check_both('EL-H3', 8.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: the cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash "$H/book_pod.sh" "$LOG" EL-H3 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
