#!/usr/bin/env bash
# EL-F2 launcher (copy of launch.sh checking el-f2-dev3, made while launch.sh was running a pod): one guarded A100 80GB pod per stage (EL-F1's pattern), booked to evalaware_lens/ledger.jsonl as EL-F2.
#   bash evalaware_lens/f2_panel/launch.sh 1     # stage 1, MAXMIN 150: MAXCOST $4.50, bound $5.40
#   bash evalaware_lens/f2_panel/launch.sh 2     # stage 2 (needs results/stage1/dirs_*.npz), MAXMIN 150
#   EL_F2_DETACH=1 ...   infra/launch_run.py without launch_wrap (a laptop memory cleanup then leaves the pod running)
#   EL_F2_LAUNCH_DRY=1 ...   print the command only
# Refuses unless the tag el-f2-prereg exists, the stage's code matches the tag (stage 2: the tag el-f2-stage2),
# the jobs match inputs_sha256.json, and LG.check_both("EL-F2", 17, planned = the bound, kind="gpu") passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/f2_panel
STAGE=${1:?usage: launch.sh 1|2}
DRY=${EL_F2_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then echo "[el-f2-launch] refusing under pytest"; exit 2; fi
export RUNPOD_ANY_CUDA=1
case "$STAGE" in
  1) NAME=el-f2-s1; RES=$H/results/stage1${RES_SUFFIX:-}; TAG=el-f2-dev3; CODE=("$H/gen.py" "$H/read.py"); EXTRA=();;
  2) NAME=el-f2-s2; RES=$H/results/stage2${RES_SUFFIX:-}; TAG=el-f2-dev3; CODE=("$H/steer.py")
     EXTRA=(); for f in "$H"/results/stage1/dirs_*.npz; do [ -f "$f" ] && EXTRA+=(--sync-extra "$f"); done
     [ ${#EXTRA[@]} -gt 0 ] || { echo "[el-f2-launch] no results/stage1/dirs_*.npz: run dirs.py first"; exit 2; };;
  *) echo "stage must be 1 or 2"; exit 2;;
esac
MAXMIN=${MAXMIN:-150}; RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
SHIP=()
for p in "${CODE[@]}" "$H/pod.sh" "$H/inputs_sha256.json" "$H/jobs" vigilante/pod/setup_vllm.sh; do SHIP+=(--ship "$p"); done
LAUNCHER=persona_affect/common/launch_wrap.py; [ -n "${EL_F2_DETACH:-}" ] && LAUNCHER=infra/launch_run.py
CMD=(uv run python "$LAUNCHER" --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" --cloud COMMUNITY --container-gb 160 "${SHIP[@]}" --exclude '__pycache__'
     ${EXTRA[@]+"${EXTRA[@]}"} --results-dir "$RES"
     -- env STAGE="$STAGE" T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" MODELS="${MODELS:-q9 q27 q32}" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then echo "[el-f2-launch] dry run:"; echo "${CMD[*]}"; exit 0; fi
[ -n "$HF_TOKEN" ] || { echo "[el-f2-launch] no HF_TOKEN in .env"; exit 2; }
git rev-parse -q --verify "refs/tags/$TAG" > /dev/null 2>&1 || { echo "[prereg] no tag $TAG"; exit 2; }
git diff --quiet "$TAG" -- "${CODE[@]}" "$H/pod.sh" "$H/DESIGN.md" || { echo "[prereg] the stage's code differs from the tag $TAG"; exit 2; }
python3 -I -c "
import gzip, hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/inputs_sha256.json').read_text())
bad = [n for n, s in want.items() if hashlib.sha256(gzip.open(f'$H/jobs/{n}.gz').read()).hexdigest() != s]
sys.exit(f'[el-f2-launch] jobs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash evalaware_lens/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c "import sys; from evalaware_lens.common import ledger as LG; LG.check_both('EL-F2', 17.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: EL-F2's cap, the programme cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash evalaware_lens/common/book_pod.sh "$LOG" EL-F2 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
