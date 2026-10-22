#!/usr/bin/env bash
# CB pod (capability belief after A1 histories; pools in inputs/): one model ($MODEL) into results/$DIR. Rows are written per chunk, so a time-capped
# pod keeps what is done (re-running `cb run` on the same dir generates only the missing ids).
# Local dry runs: PY=<python> SKIP_SETUP=1 with a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/weirdchat/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0 A1_CHUNK=${A1_CHUNK:-512}
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[cb] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.weirdchat.cb run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[cb] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
