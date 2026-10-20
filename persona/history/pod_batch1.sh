#!/usr/bin/env bash
# Persona-as-history batch 1 pod: one model ($MODEL) into results/$DIR.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/history/results/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[hist1] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.history.batch1 run --model ${MODEL:?} --out $OUT 2>&1 | tee $OUT/run.log
echo "[hist1] pod done" | tee -a $OUT/run.log
