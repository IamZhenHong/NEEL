#!/usr/bin/env bash
# Pod side of persona_affect/common/launch_gen.sh: one open model ($MODEL) on a jobs.jsonl through vigilante's vLLM runner,
# with the minis' case modules ($REGISTER) imported first (pod_gen.py). Writes episodes_<group>.jsonl and settings.json into
# $OUT, plus setup_vllm.log and run.log. Exits with the run's code.
#   env: JOBS OUT (required, repo-relative) . MODEL (Qwen/Qwen3.6-27B) FMT (qwen; gemma4 for Gemma 4) MML (40960)
#        REGISTER (space-separated modules) . SKIP_SETUP (set: vLLM venv already built) . PY (/root/vllm/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${JOBS:?set JOBS}" "${OUT:?set OUT}"
MODEL=${MODEL:-Qwen/Qwen3.6-27B} FMT=${FMT:-qwen} MML=${MML:-40960}
mkdir -p "$OUT"
# fail before the slow vLLM setup if the jobs file did not arrive (the launcher's sync drops any path that has a
# results/, logs/, prep/ or reference/ directory in it)
[ -f "$JOBS" ] || { echo "[gen] JOBS not found on the pod: $JOBS" | tee -a "$OUT/run.log"; exit 2; }
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log"; echo "[gen] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona_affect.common.pod_gen --register ${REGISTER:-} -- --jobs "$JOBS" --out "$OUT" \
  --model "$MODEL" --fmt "$FMT" --max-model-len "$MML" 2>&1 | tee "$OUT/run.log"; rc=${PIPESTATUS[0]}
echo "[gen] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
