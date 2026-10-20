#!/usr/bin/env bash
# Generic mini-0 pod: one model ($MODEL) on one jobs file ($JOBS) into results/$DIR.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=persona/exemplar; OUT=$E/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[pod] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.exemplar.mini0 run --model ${MODEL:?} --jobs ${JOBS:?} --out $OUT 2>&1 | tee $OUT/run.log
echo "[pod] done" | tee -a $OUT/run.log
