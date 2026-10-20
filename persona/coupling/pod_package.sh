#!/usr/bin/env bash
# Coupling package pod (PLAN.md "Package"): #1 + #3 through the vigilante engine, then #2 + #4 single-turn.
set -uo pipefail
cd "$(dirname "$0")/../.."
C=persona/coupling; OUT=$C/results/package; AD=$C/results/adapters
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
for m in B_s1 B_s2 C_s1 C_s2; do test -s $AD/$m/adapter_model.safetensors || { echo "[cplp] $m missing"; exit 1; }; done
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[cplp] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.coupling.stage0 run --jobs $C/jobs/package.jsonl --adapter-dir $AD --out $OUT 2>&1 | tee $OUT/run.log
/root/vllm/bin/python -m persona.coupling.package run_transfer --adapter-dir $AD --out $OUT 2>&1 | tee -a $OUT/run.log
echo "[cplp] pod done" | tee -a $OUT/run.log
