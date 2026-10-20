#!/usr/bin/env bash
# Battery calibration + #3-5 pod: one Qwen model ($MODEL), jobs file $JOBS (built locally), into
# persona/operative/battery/results/$DIR. Local dry runs only: PY=<python> SKIP_SETUP=1 with a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/battery/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[batcal] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.battery.calib run --model ${MODEL:?} --jobs ${JOBS:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[batcal] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
