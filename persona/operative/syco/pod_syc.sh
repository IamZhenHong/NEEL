#!/usr/bin/env bash
# Sycophancy pod: one Qwen model ($MODEL), STAGE=cal (jobs file $JOBS, built locally) or STAGE=main (built in-process;
# needs persona/operative/syco/policy.json from calibration, or POLICY=plain|own|insistent for dry runs), into
# persona/operative/syco/results/$DIR. Local dry runs: PY=<python> SKIP_SETUP=1 with a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/syco/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[syco] vLLM setup FAILED"; exit 1; }
fi
case "${STAGE:?STAGE must be cal or main}" in
  cal) ARGS="--jobs ${JOBS:?}" ;;
  main) ARGS="${POLICY:+--policy $POLICY}" ;;
  *) echo "STAGE must be cal or main"; exit 2 ;;
esac
${PY:-/root/vllm/bin/python} -m persona.operative.syco.syco run --stage $STAGE --model ${MODEL:?} --out $OUT $ARGS 2>&1 | tee $OUT/run_$STAGE.log; rc=${PIPESTATUS[0]}
echo "[syco] pod done (exit $rc)" | tee -a $OUT/run_$STAGE.log
exit $rc
