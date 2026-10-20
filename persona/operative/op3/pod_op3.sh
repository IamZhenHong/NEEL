#!/usr/bin/env bash
# op3 pod: one Qwen model ($MODEL) into persona/operative/op3/results/$DIR; OP3_GROUPS (comma list) optional, default
# all. SKIP_SETUP / PY for local dry runs.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/op3/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[op3] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.op3.op3 run --model ${MODEL:?} --out $OUT ${OP3_GROUPS:+--groups $OP3_GROUPS} 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[op3] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
