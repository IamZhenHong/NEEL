#!/usr/bin/env bash
# History Tier-2 MX (mixed meaningful + meaningless records) pod: one Qwen model ($MODEL) into results/$DIR.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[mx] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.history.t2_mx run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[mx] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
