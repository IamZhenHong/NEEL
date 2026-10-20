#!/usr/bin/env bash
# Mini 0e pod: restraint over a session, Qwen3.5-9B, 5,376 sessions.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=persona/exemplar; OUT=$E/results/mini0e
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[mini0e] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.exemplar.mini0e run --out $OUT 2>&1 | tee $OUT/run.log
echo "[mini0e] pod done" | tee -a $OUT/run.log
