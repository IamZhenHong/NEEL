#!/usr/bin/env bash
# Q11 + Q12b (DESIGN.md) on Qwen3.6-27B into persona/reasoning/${DIR:?}
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/reasoning/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[q11] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.reasoning.q11 run --out $OUT --groups ${QGROUPS:-A,D,det,B,C} 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[q11] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
