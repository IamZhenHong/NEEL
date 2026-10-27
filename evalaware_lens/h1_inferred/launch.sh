#!/usr/bin/env bash
# EL-H1 launcher: one guarded 2 x A100 SXM4 80GB pod (Qwen3-32B, TP 2), booked to evalaware_lens/ledger.jsonl.
#   bash evalaware_lens/h1_inferred/launch.sh        # MAXMIN 120: MAXCOST $7.16 (2 x $1.79/hr), bound $8.95
#   EL_H1_LAUNCH_DRY=1 ...                           # print the command only
# L2/L3's pattern (EL-W2's lessons): SXM4 only, MAXCOST priced at 358 c/hr (EL_H1_RATE overrides), infra/launch_run.py
# called directly (not launch_wrap), so a laptop memory-cleanup TERM leaves the pod and its setsid job running for the
# manual ssh pull. Ships only what gen.py / read.py import plus setup_vllm.sh; F2's q32 directions go as --sync-extra.
# Refuses unless the tag el-h1-prereg exists and the code matches it, the jobs match jobs_sha256.json, and
# LG.check_both passes for EL-H1 ($12) with the bound plus the judges' cap.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/h1_inferred
DRY=${EL_H1_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[el-h1-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=el-h1; RES=$H/results; MAXMIN=${MAXMIN:-120}; RATE=${EL_H1_RATE:-358}
case "$MAXMIN$RATE" in *[!0-9]*) echo "MAXMIN and EL_H1_RATE must be whole numbers"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
SHIP=()
for p in evalaware_lens/__init__.py evalaware_lens/common/__init__.py evalaware_lens/common/cues.py "$H" \
         evalaware_lens/l2l3_cuereads/__init__.py evalaware_lens/l2l3_cuereads/read.py evalaware_lens/l2l3_cuereads/items.py \
         evalaware_lens/l2l3_cuereads/jobs/sha256.json evalaware_lens/f2_panel/__init__.py evalaware_lens/f2_panel/read.py \
         vigilante/pod/setup_vllm.sh; do SHIP+=(--ship "$p"); done
EXTRA=(--sync-extra evalaware_lens/f2_panel/results/stage1/dirs_q32.npz)
EXCL=(--exclude 'data/adapters*' --exclude "$H/results" --exclude "$H/data/triggers.json" --exclude '__pycache__')
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpu-count 2 --gpus "NVIDIA A100-SXM4-80GB" --cloud COMMUNITY --container-gb 200 "${SHIP[@]}" "${EXCL[@]}"
     "${EXTRA[@]}" --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[el-h1-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
[ -n "$HF_TOKEN" ] || { echo "[el-h1-launch] no HF_TOKEN in .env"; exit 2; }
TAG=el-h1-dev1   # deviation 1 (was el-h1-prereg)
git rev-parse -q --verify "refs/tags/$TAG" > /dev/null 2>&1 || { echo "[prereg] no tag $TAG"; exit 2; }
git diff --quiet "$TAG" -- "$H" ':!*/results*' || { echo "[prereg] refusing: $H differs from $TAG"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
H = Path('$H'); want = json.loads((H / 'jobs_sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256((H / p).read_bytes()).hexdigest() != s]
sys.exit(f'[el-h1-launch] inputs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash "$H/book_pod.sh" --bound "$MAXMIN" "$MAXCOST")
uv run python -c "
import sys; from evalaware_lens.common import ledger as LG; from evalaware_lens.h1_inferred.judge import CAP, JUDGE_CAP
LG.check_both('EL-H1', CAP, planned=float(sys.argv[1]) + JUDGE_CAP, kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: a cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash "$H/book_pod.sh" "$LOG" EL-H1 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
