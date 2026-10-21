#!/usr/bin/env bash
# Tier-2 DC (defence-combination matrix) pod: one model ($MODEL) into results/$DIR.
# Decisions stop themselves DC_DEADLINE seconds into the pod script (MAXMIN - 7 min) and keep what is done.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/${DIR:?}
mkdir -p $OUT
export DC_DEADLINE=$(( $(date +%s) + (${MAXMIN:?} - 7) * 60 ))
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[dc] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.history.t2_dc run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[dc] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
