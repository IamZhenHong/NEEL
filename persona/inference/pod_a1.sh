#!/usr/bin/env bash
# A1 (identity-reminder dissociation) pod: one open model ($MODEL) into persona/inference/results/$DIR.
# Local dry runs: PY=<python> SKIP_SETUP=1 FAKE=1 with a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/inference/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[a1] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.inference.a1 run --model ${MODEL:?} --out $OUT --mode ${MODE:-full} ${FAKE:+--fake} 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[a1] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
