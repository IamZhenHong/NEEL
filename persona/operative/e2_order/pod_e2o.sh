#!/usr/bin/env bash
# E2-order pod: one model ($MODEL) into persona/operative/e2_order/results/$DIR. Decisions stop themselves
# E2O_DEADLINE seconds into this script (MAXMIN - 10 min: the launcher's job limit counts from pod creation) and keep
# whole samples. E2O_HASH (the local build's prompt hash) aborts the run if the shipped build differs.
# Local dry runs: PY=<python> SKIP_SETUP=1 with a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/e2_order/results/${DIR:?}
mkdir -p $OUT
export E2O_DEADLINE=$(( $(date +%s) + (${MAXMIN:?} - ${MARGIN:-10}) * 60 ))
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[e2o] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.e2_order.e2o run --model ${MODEL:?} --out $OUT ${SAMPLES:+--samples $SAMPLES} 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[e2o] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
