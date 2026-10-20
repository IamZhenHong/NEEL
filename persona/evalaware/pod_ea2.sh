#!/usr/bin/env bash
# ea2 Qwen part: $SIZE (9b|27b) into persona/evalaware/${DIR:?}
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/evalaware/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[ea2] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.evalaware.ea2 qwen --model ${SIZE:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[ea2] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
