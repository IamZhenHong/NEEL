#!/usr/bin/env bash
# SC-D4 launcher: one guarded A100 80GB pod running persona_affect's vLLM episode runner (persona_affect/common/pod_gen.sh)
# on Qwen3.6-27B over jobs/main_q27.jsonl, with R2G's cases registered. SC-N1's launch.sh pattern; books the pod to
# self_concept/ledger.jsonl (self_concept/common/book_pod.sh, mini SC-D4).
#   bash self_concept/d4_rolerefs/launch.sh                                        # MAXMIN (60)
# Refuses unless the tag sc-d4-prereg exists, the jobs file matches its sha256 in DESIGN.md, pod_gen --check passes and
# SC-D4's ledger + this pod's upper bound stay within $5 (owner raised the cap from $3 on 10-23 after the first pod's stall; the judges then
# get what is left, at most $1.50). D4_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/d4_rolerefs
DRY=${D4_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[d4-launch] refusing to launch a pod under pytest (D4_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=sc-d4-q27; JOBS=$H/jobs/main_q27.jsonl; OUT=$H/results/main_q27; MAXMIN=${MAXMIN:-80}
MODEL=Qwen/Qwen3.6-27B; REGISTER=persona_affect.r2g_defence.cases
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
EXCL=()
for d in persona_affect/*/jobs persona_affect/*/data/adapters*; do [ -d "$d" ] && EXCL+=(--exclude "$d"); done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 120 --cloud COMMUNITY
     --ship persona_affect --ship vigilante/src --ship vigilante/cases --ship vigilante/pod --ship vigilante/__init__.py
     --ship infra --ship self_concept/__init__.py --ship "$H" --exclude "$H/results" --exclude 'data/adapters*' "${EXCL[@]}" --results-dir "$OUT"
     -- env MODEL="$MODEL" JOBS="$JOBS" OUT="$OUT" REGISTER="$REGISTER" FMT=qwen MML=40960 bash persona_affect/common/pod_gen.sh)
if [ "$DRY" = 1 ]; then
  echo "[d4-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-d4-prereg > /dev/null 2>&1 || { echo "[prereg] no tag sc-d4-prereg"; exit 2; }
sha=$(shasum -a 256 "$JOBS" | cut -d' ' -f1)
grep -q "$sha" "$H/DESIGN.md" || { echo "[d4-launch] $JOBS sha256 $sha is not in $H/DESIGN.md"; exit 2; }
uv run python -m persona_affect.common.pod_gen --register "$REGISTER" --check "$JOBS" --fmt qwen \
  || { echo "[pre] refusing to launch: pod_gen --check failed"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c 'import sys; from self_concept.common.ledger import check_cap; check_cap("SC-D4", 5.0, planned=float(sys.argv[1]))' "$BOUND" \
  || { echo "[budget] refusing: SC-D4's ledger + this pod's bound \$$BOUND would pass \$5"; exit 2; }
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-D4 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
