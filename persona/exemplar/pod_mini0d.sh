#!/usr/bin/env bash
# Mini 0d pod: one model ($MODEL) on the 11,520 0d items.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=persona/exemplar; OUT=$E/results/${MINI0D_DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[mini0d] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.exemplar.mini0 run --model ${MODEL:?} --jobs $E/jobs/mini0d.jsonl --out $OUT 2>&1 | tee $OUT/run.log
echo "[mini0d] pod done" | tee -a $OUT/run.log
