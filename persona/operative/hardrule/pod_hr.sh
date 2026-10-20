#!/usr/bin/env bash
# Mini 3 pod: one Qwen model ($MODEL) into persona/operative/hardrule/results/$DIR. SKIP_SETUP / PY for local dry runs.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/hardrule/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[hr] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.hardrule.hr run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[hr] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
