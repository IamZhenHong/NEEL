#!/usr/bin/env bash
# EL-S1 deviation 1 (DESIGN.md): a second generation pod for one cell, running the frozen jobs its first pod didn't
# reach (same jobs, same seeds, same sampling). It is a new file because launch.sh was running when this was written
# (never edit a running launch script).
#   CELL=Q27 bash evalaware_lens/s1_selfref/launch_gen2.sh        # 2 x A100 SXM4 (TP 2), MAXMIN 90
#   EL_S1_LAUNCH_DRY=1 CELL=Q27 ...                                # print the command only
# What it does:
# - The skip list is every job key the first pod's pulled records hold (data/gen/<cell>/eval_*.jsonl[.gz]), written
#   to data/gen2/skip_<cell>.txt and shipped.
# - Results go to data/gen/<cell>b, which judge.gen_records reads with <cell>'s.
# - Booked by book_split.py (any Fortress jobs to EL-L1, by generation time).
# Refuses unless:
# - the tag el-s1-dev1 exists and the code matches it;
# - the jobs match jobs_sha256.json;
# - check_both passes for EL-L1 and EL-S1.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/s1_selfref
DRY=${EL_S1_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[el-s1-gen2] refusing to launch a pod under pytest"; exit 2
fi
case "${CELL:-}" in Q27|Q32) ;; *) echo "set CELL=Q27 or CELL=Q32"; exit 2;; esac
NGPU=2; RATE=360; DISK=160; MAXMIN=${MAXMIN:-90}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
GPUS=("NVIDIA A100-SXM4-80GB")
export RUNPOD_ANY_CUDA=1
LOWER=$(echo "$CELL" | tr '[:upper:]' '[:lower:]')
NAME=el-s1-gen2-$LOWER; RES=$H/data/gen/${CELL}b; LOG=$H/results/gen2.$CELL.launch.log
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
SHIP=(--ship evalaware_lens/__init__.py --ship "$H" --ship vigilante/pod/setup_vllm.sh)
EXTRA=(--exclude "$H/data/glm" --exclude "$H/data/fortress" --exclude "$H/data/am_inspect" --exclude "$H/data/gen"
       --exclude "$H/data/labels" --exclude "$H/data/edits" --exclude "$H/results" --exclude '__pycache__'
       --exclude 'data/adapters*')
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpu-count "$NGPU" --gpus "${GPUS[@]}" --cloud COMMUNITY --container-gb "$DISK"
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" CELL="$CELL" STAGE=gen2 NGPU="$NGPU" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[el-s1-gen2] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
[ -n "$HF_TOKEN" ] || { echo "[el-s1-gen2] no HF_TOKEN in .env"; exit 2; }
TAG=el-s1-dev2
git rev-parse -q --verify "refs/tags/$TAG" > /dev/null 2>&1 \
  || { echo "[prereg] refusing: no tag $TAG"; exit 2; }
git diff --quiet "$TAG" -- "$H" ':!*/results/*' \
  || { echo "[prereg] refusing: $H differs from the tag $TAG"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
from evalaware_lens.s1_selfref import judge as JD
H = Path('$H'); want = json.loads((H / 'jobs_sha256.json').read_text())
have = {n: hashlib.sha256((H / f'data/jobs/{n}.jsonl.gz').read_bytes()).hexdigest() for n in want}
if have != want:
    sys.exit(f'[prereg] refusing: jobs differ from jobs_sha256.json: {have}')
keys = sorted({r['key'] for src in ('fortress', 'am', 'needham') for r in JD.read_jsonl(JD.GEN / '$CELL' / f'eval_{src}.jsonl')})
if not keys:
    sys.exit('[el-s1-gen2] refusing: no pulled records for $CELL; nothing to skip (use launch.sh)')
(H / 'data/gen2').mkdir(parents=True, exist_ok=True)
(H / 'data/gen2/skip_$CELL.txt').write_text('\n'.join(keys) + '\n')
print(f'[el-s1-gen2] skip list: {len(keys)} keys already generated for $CELL')"
BOUND=$(bash "$H/book_pod.sh" --bound "$MAXMIN" "$MAXCOST")
read -r PS1_ PL1 < <(python3 -c "import sys; b = float(sys.argv[1]); print(round(b * 0.8, 2), round(b * 0.2, 2))" "$BOUND")
uv run python -c "import sys; from evalaware_lens.common import ledger as LG; LG.check_both('EL-L1', 11.7, planned=float(sys.argv[1]), kind='gpu'); LG.check_both('EL-S1', 21.5, planned=float(sys.argv[2]), kind='gpu')" "$PL1" "$PS1_" \
  || { echo "[budget] refusing to launch (see above)"; exit 2; }
mkdir -p "$RES" "$H/results"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
uv run python -m evalaware_lens.s1_selfref.book_split "$LOG" "$NAME" "$MAXMIN" "$MAXCOST" "$RES" "$CELL" \
  || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
