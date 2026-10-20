#!/usr/bin/env bash
# rr2 pod: one Qwen model ($MODEL) into persona/operative/reflection_read/${DIR:?}
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/reflection_read/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[rr2] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.operative.reflection_read.rr2 run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[rr2] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
