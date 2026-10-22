#!/usr/bin/env bash
# PA-A8 pod side (persona_affect/a8_rejection/launch.sh starts it): build the vLLM venv, then generate the histories and the
# branches for one model ($M) into $OUT. Generation only; scoring and judging run locally afterwards.
#   env: M (required: g3 | gemma | q27 | l8)  OUT (results dir, repo-relative; default persona_affect/a8_rejection/results/$M)
#        LIMIT (only the first N branches of the shuffled list; the histories stay complete)
#        SKIP_SETUP (set: the vLLM venv already exists)  PY (/root/vllm/bin/python)
# Writes histories.json, branches.partial.jsonl, branches.jsonl and run_info.json into $OUT, plus setup_vllm.log and run.log.
# Exits with the run's code.
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${M:?set M (g3, gemma, q27 or l8)}"
OUT=${OUT:-persona_affect/a8_rejection/results/$M}
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log"; echo "[a8] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona_affect.a8_rejection.a8 run --model "$M" --out "$OUT" ${LIMIT:+--limit "$LIMIT"} 2>&1 | tee "$OUT/run.log"; rc=${PIPESTATUS[0]}
echo "[a8] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
