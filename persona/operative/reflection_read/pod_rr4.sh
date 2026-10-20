#!/usr/bin/env bash
# rr4 Qwen part: Qwen3.6-27B into persona/operative/reflection_read/${DIR:?}
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/reflection_read/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[rr4] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.operative.reflection_read.rr4 qwen --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[rr4] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
