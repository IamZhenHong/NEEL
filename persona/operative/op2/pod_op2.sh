#!/usr/bin/env bash
# op2 pod: one Qwen model ($MODEL) into persona/operative/op2/results/$DIR; OP2_GROUPS (comma list) optional, default
# all. SKIP_SETUP / PY for local dry runs.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/op2/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[op2] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.op2.op2 run --model ${MODEL:?} --out $OUT ${OP2_GROUPS:+--groups $OP2_GROUPS} 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[op2] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
