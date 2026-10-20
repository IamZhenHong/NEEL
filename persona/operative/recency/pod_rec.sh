#!/usr/bin/env bash
# Records-vs-recency pod: one Qwen model ($MODEL) into persona/operative/recency/results/$DIR. SKIP_SETUP / PY for local dry runs.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/recency/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[rec] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.recency.rec run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[rec] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
