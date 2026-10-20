#!/usr/bin/env bash
# persona/inference P1-b pod: one Qwen model ($MODEL) into persona/inference/results/$DIR.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/inference/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[inf] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.inference.infer run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[inf] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
