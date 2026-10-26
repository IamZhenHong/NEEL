#!/usr/bin/env bash
# EL-S1 launcher (DESIGN.md section 4): one guarded pod per call, booked to evalaware_lens/ledger.jsonl.
#   STAGE=gen CELL=Q27 bash evalaware_lens/s1_selfref/launch.sh   # 1 x A100 SXM4, MAXMIN 120
#   STAGE=gen CELL=Q32 bash evalaware_lens/s1_selfref/launch.sh   # 2 x A100 SXM4 (TP 2), MAXMIN 110
#   STAGE=cont CELL=Q27|Q32 ...                                   # amendment 1's fortress_para, then the edit test:
#                                                                 #   2 x A100 (TP 2) both cells; MAXMIN Q27 100, Q32 75
#   EL_S1_LAUNCH_DRY=1 ...                                        # print the command only
# Gen pods are booked split between EL-L1 and EL-S1 by book_split.py (fortress -> EL-L1); cont pods go to EL-S1.
# Ships the code, the jobs and setup_vllm.sh; the bases download on the pod. HF_TOKEN is read from .env and never
# printed.
# Refuses unless:
# - the tag el-s1-prereg exists and the code matches it;
# - the jobs match jobs_sha256.json (gen), or the edit jobs match data/edits/sha256 (cont);
# - check_both passes: EL-S1 $14 for S1's planned share, and EL-L1 for the fortress share of a gen pod.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/s1_selfref
DRY=${EL_S1_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[el-s1-launch] refusing to launch a pod under pytest"; exit 2
fi
case "${STAGE:-}" in gen|cont) ;; *) echo "set STAGE=gen or STAGE=cont"; exit 2;; esac
case "${CELL:-}" in
  Q27) NGPU=1; RATE=180; DISK=160 ;;
  Q32) NGPU=2; RATE=360; DISK=180 ;;
  *) echo "set CELL=Q27 or CELL=Q32"; exit 2 ;;
esac
if [ "$STAGE" = gen ]; then DEF=$([ "$CELL" = Q27 ] && echo 120 || echo 110); else DEF=$([ "$CELL" = Q27 ] && echo 100 || echo 75); fi
# deviation 2: the cont stage runs on 2 x A100 for both cells (Q27 is ~3x slower on one card, deviation 1)
if [ "$STAGE" = cont ]; then NGPU=2; RATE=360; fi
MAXMIN=${MAXMIN:-$DEF}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
GPUS=("NVIDIA A100-SXM4-80GB")   # SXM4 only: the community PCIe host 104.255.9.187 fails uv downloads (10-25)
export RUNPOD_ANY_CUDA=1
LOWER=$(echo "$CELL" | tr '[:upper:]' '[:lower:]')
NAME=el-s1-$STAGE-$LOWER; RES=$H/data/gen/$CELL; LOG=$H/results/$STAGE.$CELL.launch.log
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
SHIP=(--ship evalaware_lens/__init__.py --ship "$H" --ship vigilante/pod/setup_vllm.sh)
EXTRA=(--exclude "$H/data/glm" --exclude "$H/data/fortress" --exclude "$H/data/am_inspect" --exclude "$H/data/gen"
       --exclude "$H/data/labels" --exclude "$H/results" --exclude '__pycache__' --exclude 'data/adapters*')
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpu-count "$NGPU" --gpus "${GPUS[@]}" --cloud COMMUNITY --container-gb "$DISK"
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" CELL="$CELL" STAGE="$STAGE" NGPU="$NGPU" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[el-s1-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
[ -n "$HF_TOKEN" ] || { echo "[el-s1-launch] no HF_TOKEN in .env"; exit 2; }
TAG=el-s1-dev2   # deviation 2 (was el-s1-prereg)
git rev-parse -q --verify "refs/tags/$TAG" > /dev/null 2>&1 \
  || { echo "[prereg] refusing: no tag $TAG; commit and tag DESIGN.md and the code first"; exit 2; }
git diff --quiet "$TAG" -- "$H" ':!*/results/*' \
  || { echo "[prereg] refusing: $H differs from the tag $TAG"; exit 2; }
if [ "$STAGE" = gen ]; then
  uv run python -c "
import hashlib, json, sys; from pathlib import Path
H = Path('$H'); want = json.loads((H / 'jobs_sha256.json').read_text())
have = {n: hashlib.sha256((H / f'data/jobs/{n}.jsonl.gz').read_bytes()).hexdigest() for n in want}
same = json.loads((H / 'data/jobs/sha256.json').read_text()) == want
sys.exit(0 if have == want and same else f'[prereg] refusing: jobs differ from jobs_sha256.json: {have}')"
else
  [ -s "$H/data/edits/sha256" ] && [ "$(shasum -a 256 "$H/data/edits/cont_jobs.jsonl.gz" | cut -d' ' -f1)" = "$(cut -d' ' -f1 "$H/data/edits/sha256")" ] \
    || { echo "[el-s1-launch] refusing: data/edits/cont_jobs.jsonl.gz does not match data/edits/sha256"; exit 2; }
fi
BOUND=$(bash "$H/book_pod.sh" --bound "$MAXMIN" "$MAXCOST")
if [ "$STAGE" = gen ]; then
  read -r PS1_ PL1 < <(python3 -c "import sys; b = float(sys.argv[1]); print(round(b * 0.4, 2), round(b * 0.6, 2))" "$BOUND")
  uv run python -c "import sys; from evalaware_lens.common import ledger as LG; LG.check_both('EL-L1', 11.7, planned=float(sys.argv[1]), kind='gpu'); LG.check_both('EL-S1', 21.5, planned=float(sys.argv[2]), kind='gpu')" "$PL1" "$PS1_" \
    || { echo "[budget] refusing to launch (see above)"; exit 2; }
else
  uv run python -c "import sys; from evalaware_lens.common import ledger as LG; LG.check_both('EL-S1', 21.5, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
    || { echo "[budget] refusing to launch (see above)"; exit 2; }
fi
mkdir -p "$RES" "$H/results"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
if [ "$STAGE" = gen ]; then
  uv run python -m evalaware_lens.s1_selfref.book_split "$LOG" "$NAME" "$MAXMIN" "$MAXCOST" "$RES" "$CELL" \
    || echo "[ledger] the booking step failed (see above)"
else
  bash "$H/book_pod.sh" "$LOG" EL-S1 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
fi
trap - INT TERM HUP
exit "$rc"
