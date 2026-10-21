#!/usr/bin/env bash
# Tier-2 DA pod (default assistant by precedent): vLLM own-text elicitation + probe replies, then HF assistant axis +
# context residuals, into results/$DIR. The activation stage stops itself DA_DEADLINE seconds into the pod script
# (MAXMIN - 12 min) and saves what it has; the replies (the primary) are written before it starts.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/${DIR:?}
mkdir -p $OUT
export DA_DEADLINE=$(( $(date +%s) + (${MAXMIN:-55} - 12) * 60 ))
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[da] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.history.t2_da gen --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run_gen.log; rc=${PIPESTATUS[0]}
if [ $rc -eq 0 ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/venv/bin/python -m persona.history.t2_da acts --model $MODEL --out $OUT 2>&1 | tee $OUT/run_acts.log; rc=${PIPESTATUS[0]}
fi
echo "[da] pod done (exit $rc)" | tee -a $OUT/run_gen.log
exit $rc
