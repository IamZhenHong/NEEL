#!/usr/bin/env bash
# Gemma 4 31B pod: history batches 1, 2, 3 into results/gemma/{b1,b2,b3}.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/gemma
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[gemma] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.history.gemma_run --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[gemma] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
