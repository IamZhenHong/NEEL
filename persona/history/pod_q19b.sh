#!/usr/bin/env bash
# History Q19 follow-up pod (q19b): one Qwen model ($MODEL) into results/$DIR.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[q19b] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.history.q19b run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[q19b] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
