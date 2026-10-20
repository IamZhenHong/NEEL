#!/usr/bin/env bash
# Re-injection schedule pod: one Qwen model ($MODEL) into persona/operative/reinject/results/$DIR.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/reinject/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[rij] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.reinject.reinject run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[rij] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
