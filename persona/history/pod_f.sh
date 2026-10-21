#!/usr/bin/env bash
# Run F pod (first-thinking-token probe, Qwen3.6-27B): vLLM thinking traces, then HF activations, into results/$DIR.
# The activation stage stops itself F_DEADLINE seconds into the pod script (MAXMIN - 18 min) and saves what it has.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/${DIR:?}
mkdir -p $OUT
export F_DEADLINE=$(( $(date +%s) + (${MAXMIN:-125} - 18) * 60 ))
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[f] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.history.f_probe gen --out $OUT 2>&1 | tee $OUT/run_gen.log; rc=${PIPESTATUS[0]}
if [ $rc -eq 0 ]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /root/venv/bin/python -m persona.history.f_probe acts --out $OUT 2>&1 | tee $OUT/run_acts.log; rc=${PIPESTATUS[0]}
fi
echo "[f] pod done (exit $rc)" | tee -a $OUT/run_gen.log
exit $rc
