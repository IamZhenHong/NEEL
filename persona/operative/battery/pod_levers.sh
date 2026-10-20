#!/usr/bin/env bash
# Battery lever runs pod (battery/DESIGN.md "Lever runs"): Qwen3.6-27B into persona/operative/battery/results/$DIR.
# SKIP_SETUP=1 / PY=<python> let a local dry run use a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/operative/battery/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[batlev] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.operative.battery.levers run --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[batlev] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
