#!/usr/bin/env bash
# observer pod: one model ($MODEL), 3 records x 4 framings x 3 readouts x 16 targets x 8 samples, into
# persona/operative/observer/results/$DIR. Local dry runs: PY=<python> SKIP_SETUP=1 with a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/observer/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[obs] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.observer.observer run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[obs] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
