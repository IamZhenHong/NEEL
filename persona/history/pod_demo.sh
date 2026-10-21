#!/usr/bin/env bash
# Run A pod (demonstrations vs own record): arm J for $MODEL, then arm G when MODEL is the 27B, into results/$DIR.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[demo] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.history.demo run_j --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
if [ $rc -eq 0 ] && [ "$MODEL" = "Qwen/Qwen3.6-27B" ]; then
  /root/vllm/bin/python -m persona.history.demo run_g --out $OUT/g 2>&1 | tee $OUT/run_g.log; rc=${PIPESTATUS[0]}
fi
echo "[demo] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
