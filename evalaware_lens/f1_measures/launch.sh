#!/usr/bin/env bash
# EL-F1 launcher: one guarded A100 80GB pod (SC-M30 / EL-W2's pattern), booked to evalaware_lens/ledger.jsonl as EL-F1.
#   bash evalaware_lens/f1_measures/launch.sh             # MAXMIN 240: MAXCOST $7.20, bound $8.10
#   EL_F1_DETACH=1 bash evalaware_lens/f1_measures/launch.sh   # infra/launch_run.py without launch_wrap (EL-W2's fix:
#       a laptop memory-cleanup TERM then leaves the pod and its setsid job running, for the manual ssh pull)
#   EL_F1_LAUNCH_DRY=1 ...                                 # print the command only
# Ships the code, the frozen jobs (data/jobs, gitignored) and setup_vllm.sh; never Needham's raw dataset or results/.
# Refuses unless the tag el-f1-prereg exists, the code matches it, the jobs match inputs_sha256.json and
# LG.check_both("EL-F1", 11, planned = the bound, kind="gpu") passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/f1_measures
DRY=${EL_F1_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[el-f1-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=el-f1; RES=$H/results/main
MAXMIN=${MAXMIN:-240}; RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
SHIP=()
for p in "$H/gen.py" "$H/read.py" "$H/pod.sh" "$H/inputs_sha256.json" "$H/data/jobs" vigilante/pod/setup_vllm.sh; do SHIP+=(--ship "$p"); done
EXCL=(--exclude '__pycache__')
LAUNCHER=persona_affect/common/launch_wrap.py; [ -n "${EL_F1_DETACH:-}" ] && LAUNCHER=infra/launch_run.py
CMD=(uv run python "$LAUNCHER" --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" --cloud COMMUNITY --container-gb 200 "${SHIP[@]}" "${EXCL[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[el-f1-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
[ -n "$HF_TOKEN" ] || { echo "[el-f1-launch] no HF_TOKEN in .env"; exit 2; }
git rev-parse -q --verify refs/tags/el-f1-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag el-f1-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
git diff --quiet el-f1-prereg -- "$H" ':!*/results/*' ':!*/data/*' \
  || { echo "[prereg] $H differs from the tag el-f1-prereg"; exit 2; }
python3 -I -c "
import gzip, hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/inputs_sha256.json').read_text())
bad = [n for n, s in want.items() if hashlib.sha256(gzip.open(f'$H/data/jobs/{n}.gz').read()).hexdigest() != s]
sys.exit(f'[el-f1-launch] jobs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash evalaware_lens/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c "import sys; from evalaware_lens.common import ledger as LG; LG.check_both('EL-F1', 11.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: EL-F1's cap, the programme cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash evalaware_lens/common/book_pod.sh "$LOG" EL-F1 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
