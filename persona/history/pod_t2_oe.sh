#!/usr/bin/env bash
# T2-OE (open-ended actions judged for character fit) pod: one model ($MODEL) into results/$DIR.
# Generation stops itself between chunks OE_DEADLINE seconds into the pod script (MAXMIN - 15 min) and keeps what it has.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/${DIR:?}
mkdir -p $OUT
export OE_DEADLINE=$(( $(date +%s) + (${MAXMIN:-55} - 15) * 60 ))
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[oe] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.history.t2_oe run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[oe] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
