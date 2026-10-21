#!/usr/bin/env bash
# Tier-2 PU pod (purity-checked value / belief / reasoning histories): one model ($MODEL) into results/$DIR.
# Decisions run first; the run stops itself at MAXMIN - 8 min (T2PU_DEADLINE) and keeps what is done.
# Local dry runs: PY=<python> SKIP_SETUP=1 with a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/${DIR:?}
mkdir -p $OUT
export T2PU_DEADLINE=$(( $(date +%s) + (${MAXMIN:-60} - 8) * 60 ))
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[t2pu] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.history.t2_pu run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[t2pu] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
