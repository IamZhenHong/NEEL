#!/usr/bin/env bash
# B2 / B8 / B9 (DESIGN.md) on Qwen $SIZE into persona/benchmarks/${DIR:?}
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/benchmarks/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[bench] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.benchmarks.bench run --model ${SIZE:?} --out $OUT 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[bench] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
