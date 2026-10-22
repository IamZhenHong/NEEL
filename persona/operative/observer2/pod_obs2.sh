#!/usr/bin/env bash
# observer2 pod: one model ($MODEL), the 2x2's new cells (3 records x 5 framings x 2 readouts x 16 targets x 8 samples)
# + a 96-item determinism check of obs items, into persona/operative/observer2/results/$DIR. Local dry runs:
# PY=<python> SKIP_SETUP=1 with a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/observer2/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[obs2] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.observer2.observer2 run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[obs2] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
