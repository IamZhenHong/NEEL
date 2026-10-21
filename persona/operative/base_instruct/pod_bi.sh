#!/usr/bin/env bash
# Base vs instruct pod: one pair ($PAIR: gemma | qwen9), pt first (G2 before any comparison output), then it, into
# persona/operative/base_instruct/results/$PAIR.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/base_instruct/results/${PAIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[bi] vLLM setup FAILED"; exit 1; }
fi
# pt then it, each in its own process (frees the GPU); the it step exits at once if the pt model failed G2
${PY:-/root/vllm/bin/python} -m persona.operative.base_instruct.bi run --pair $PAIR --kind pt --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
if [ $rc -eq 0 ]; then
  ${PY:-/root/vllm/bin/python} -m persona.operative.base_instruct.bi run --pair $PAIR --kind it --out $OUT 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
fi
echo "[bi] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
