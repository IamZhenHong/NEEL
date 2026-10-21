#!/usr/bin/env bash
# Tier-2 character-consistency battery pod: one model ($MODEL) into results/$DIR. Generation stops itself at a chunk
# boundary CC_DEADLINE seconds into the pod script (MAXMIN - 12 min) and keeps what is done (sample-major order).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/${DIR:?}
mkdir -p $OUT
export CC_DEADLINE=$(( $(date +%s) + (${MAXMIN:?} - 12) * 60 ))
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[cc] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.history.t2_cc run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[cc] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
