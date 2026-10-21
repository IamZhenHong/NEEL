#!/usr/bin/env bash
# E2 pod: one model ($MODEL), all 38 cells x 16 targets x 12, into persona/operative/e2_reassert/results/$DIR.
# Local dry runs: PY=<python> SKIP_SETUP=1 with a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/e2_reassert/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[e2] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.e2_reassert.e2 run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[e2] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
