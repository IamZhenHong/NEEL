#!/usr/bin/env bash
# History Tier-2 MR (the mask in reverse) pod: one open model ($MODEL) into results/$DIR.
# Generation stops itself T2MR_DEADLINE seconds into the pod script (MAXMIN - 10 min) and keeps what it has; items run
# primary contexts first, then the forward-mask anchor, then the reminder arm.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/${DIR:?}
mkdir -p $OUT
export T2MR_DEADLINE=$(( $(date +%s) + (${MAXMIN:-45} - 10) * 60 ))
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[mr] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.history.t2_mr run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[mr] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
