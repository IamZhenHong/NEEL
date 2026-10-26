#!/usr/bin/env bash
# EL-S3 stage B launcher: one guarded pod (one A100 80GB, MAXMIN 50), booked to evalaware_lens/ledger.jsonl as EL-S3.
#   bash evalaware_lens/s3_mask/launch_b.sh
#   EL_S3B_LAUNCH_DRY=1 bash evalaware_lens/s3_mask/launch_b.sh     # print the command only
# Ships only the code stage B needs (SC-M34's axis prompts and Runner, the lens loader, axis_pod.py and its input).
# Refuses unless the tag el-s3-prereg exists, the deviation note B is committed (axis_pod.py, pod_b.sh and this file
# match HEAD), stageB_in/claims_qwen35.jsonl exists, and check_both("EL-S3", 9, planned, kind="gpu") passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/s3_mask
DRY=${EL_S3B_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[s3b-launch] refusing to launch a pod under pytest"; exit 2
fi
MAXMIN=${MAXMIN:-50}; RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=el-s3b; RES=$H/results/stageB
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=(--ship self_concept/__init__.py --ship self_concept/common/__init__.py --ship self_concept/common/lens.py
      --ship self_concept/m8_steer --ship self_concept/m34_cap --ship evalaware_lens/__init__.py
      --ship "$H/__init__.py" --ship "$H/axis_pod.py" --ship "$H/pod_b.sh" --ship "$H/stageB_in")
EXTRA=(--exclude self_concept/m8_steer/results --exclude self_concept/m34_cap/results --exclude 'data/adapters*')
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY
     --container-gb 120 "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_b.sh")
if [ "$DRY" = 1 ]; then
  echo "[s3b-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/el-s3-prereg > /dev/null 2>&1 || { echo "[prereg] no tag el-s3-prereg"; exit 2; }
git diff --quiet HEAD -- "$H/axis_pod.py" "$H/pod_b.sh" "$H/launch_b.sh" "$H/axis_analyze.py" "$H/DESIGN.md" \
  || { echo "[prereg] stage B's code or DESIGN.md differs from HEAD: commit the deviation note B first"; exit 2; }
[ -s "$H/stageB_in/claims_qwen35.jsonl" ] || { echo "[s3b] no stageB_in/claims_qwen35.jsonl: run axis_pod --build-input on stage A's Qwen run"; exit 2; }
BOUND=$(bash $H/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c 'import sys; from evalaware_lens.common import ledger as LG; LG.check_both("EL-S3", 9.0, planned=float(sys.argv[1]), kind="gpu")' "$BOUND" \
  || { echo "[budget] refusing to launch: check_both failed for EL-S3 (cap \$9, bound \$$BOUND)"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash $H/book_pod.sh "$LOG" EL-S3 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
