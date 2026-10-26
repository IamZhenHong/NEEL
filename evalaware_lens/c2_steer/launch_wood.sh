#!/usr/bin/env bash
# EL-C2 testbed B launcher: one guarded pod (2 x A100 80GB by default; GPUSET=h200 for 1 x H200), booked to
# evalaware_lens/ledger.jsonl as EL-C2.
#   bash evalaware_lens/c2_steer/launch_wood.sh
#   EL_C2_LAUNCH_DRY=1 bash evalaware_lens/c2_steer/launch_wood.sh     # print the command only
# Ships only what the pod imports: evalaware_lens/c2_steer (no results), F2's dirs.py and job files. Refuses unless the
# tag el-c2-prereg exists, C2's code and DESIGN.md match HEAD, data/items.jsonl.gz matches DESIGN.md's sha256, and
# check_both("EL-C2", 20, planned, kind="gpu") passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/c2_steer
DRY=${EL_C2_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[c2-launch] refusing to launch a pod under pytest"; exit 2
fi
MAXMIN=${MAXMIN:-200}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
case "${GPUSET:-a100x2}" in
  a100x2) GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); NGPU=2; RATE=358;;
  h200)   GPUS=("NVIDIA H200"); NGPU=1; RATE=399;;
  *) echo "GPUSET must be a100x2 or h200"; exit 2;;
esac
export RUNPOD_ANY_CUDA=1
NAME=el-c2-wood; RES=$H/results/wood
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=(--ship evalaware_lens/__init__.py --ship "$H" --ship evalaware_lens/f2_panel/__init__.py
      --ship evalaware_lens/f2_panel/dirs.py --ship evalaware_lens/f2_panel/jobs)
EXTRA=(--exclude "$H/results" --exclude 'data/adapters*')
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpu-count "$NGPU" --gpus "${GPUS[@]}" --cloud COMMUNITY --container-gb 220
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_wood.sh")
if [ "$DRY" = 1 ]; then
  echo "[c2-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/el-c2-prereg > /dev/null 2>&1 || { echo "[prereg] no tag el-c2-prereg"; exit 2; }
git diff --quiet HEAD -- "$H" ":!$H/results" ":!$H/results/*" \
  || { echo "[prereg] $H differs from HEAD: commit (a deviation note) first"; exit 2; }
WANT=$(sed -n 's/^.*items sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$H/data/items.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] items sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
BOUND=$(bash evalaware_lens/s3_mask/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c 'import sys; from evalaware_lens.common import ledger as LG; LG.check_both("EL-C2", 20.0, planned=float(sys.argv[1]), kind="gpu")' "$BOUND" \
  || { echo "[budget] refusing to launch: check_both failed for EL-C2 (cap \$20, bound \$$BOUND)"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash evalaware_lens/s3_mask/book_pod.sh "$LOG" EL-C2 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
