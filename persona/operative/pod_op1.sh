#!/usr/bin/env bash
# op1 pod: one Qwen model ($MODEL) into persona/operative/results/$DIR; OP1_GROUPS (comma list) optional, default all.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/operative/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[op1] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.operative.op1 run --model ${MODEL:?} --out $OUT ${OP1_GROUPS:+--groups $OP1_GROUPS} 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[op1] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
