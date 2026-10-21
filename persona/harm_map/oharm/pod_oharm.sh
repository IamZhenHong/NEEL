#!/usr/bin/env bash
# Explicit operational harm pod (oharm/DESIGN.md): one open model ($MODEL) into persona/harm_map/oharm/results/$DIR.
# SKIP_SETUP=1 / PY=<python> let a local dry run use a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../../.."
OUT=persona/harm_map/oharm/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[oh] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.harm_map.oharm.oharm run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[oh] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
