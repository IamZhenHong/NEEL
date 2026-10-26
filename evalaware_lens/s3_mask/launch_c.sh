#!/usr/bin/env bash
# EL-S3 stage C launcher: one guarded pod (one A100 80GB, MAXMIN 60), booked to evalaware_lens/ledger.jsonl as EL-S3.
#   bash evalaware_lens/s3_mask/launch_c.sh
#   EL_S3C_LAUNCH_DRY=1 bash evalaware_lens/s3_mask/launch_c.sh     # print the command only
# Ships only the code stage C needs (SC-S1's trainer and RISKY rows, SC-M8's Runner, SC-M24's items and attach(), the
# lens loader, stage_c.py and jobs_c). Refuses unless the tag el-s3-prereg exists, the deviation note C is committed
# (stage_c.py, pod_c.sh, c_analyze.py, this file and DESIGN.md match HEAD), jobs_c's sha256 matches DESIGN.md, and
# check_both("EL-S3", 9, planned, kind="gpu") passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/s3_mask
DRY=${EL_S3C_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[s3c-launch] refusing to launch a pod under pytest"; exit 2
fi
MAXMIN=${MAXMIN:-60}; RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=el-s3c; RES=$H/results/stageC
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=(--ship persona_affect/__init__.py --ship persona_affect/common/__init__.py --ship persona_affect/common/util.py
      --ship persona_affect/v1_value_hook/__init__.py --ship persona_affect/v1_value_hook/train.py
      --ship persona_affect/v1_value_hook/fmt.py --ship persona/operative/lora_c1e4/__init__.py
      --ship persona/operative/lora_c1e4/common.py --ship persona/operative/lora_c1e4/train.py
      --ship self_concept/__init__.py --ship self_concept/common/__init__.py --ship self_concept/common/lens.py
      --ship self_concept/s1_scale/__init__.py --ship self_concept/s1_scale/train_any.py
      --ship self_concept/s1_scale/data/rows/RISKY.jsonl.gz --ship self_concept/m8_steer
      --ship self_concept/m24_persist/__init__.py --ship self_concept/m24_persist/items.py
      --ship self_concept/m24_persist/pod.py --ship evalaware_lens/__init__.py --ship "$H/__init__.py"
      --ship "$H/stage_c.py" --ship "$H/pod_c.sh" --ship "$H/jobs_c")
EXTRA=(--exclude self_concept/m8_steer/results --exclude 'data/adapters*')
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY
     --container-gb 200 "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_c.sh")
if [ "$DRY" = 1 ]; then
  echo "[s3c-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/el-s3-prereg > /dev/null 2>&1 || { echo "[prereg] no tag el-s3-prereg"; exit 2; }
git diff --quiet HEAD -- "$H/stage_c.py" "$H/pod_c.sh" "$H/launch_c.sh" "$H/c_analyze.py" "$H/DESIGN.md" \
  || { echo "[prereg] stage C's code or DESIGN.md differs from HEAD: commit the deviation note C first"; exit 2; }
WANT=$(sed -n 's/^.*jobs_c sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$H/jobs_c/behaviour.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] jobs_c sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
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
