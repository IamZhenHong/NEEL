#!/usr/bin/env bash
# op4 pod: one model ($MODEL), phase 1 (own inducer texts) then the 12 conditions x 5 channels, into
# persona/operative/op4/results/$DIR. Local dry runs: PY=<python> SKIP_SETUP=1 with a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/op4/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[op4] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.op4.op4 run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[op4] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
