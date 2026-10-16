#!/usr/bin/env bash
# e64b addendum B-C1 pod (one A100 80GB, vLLM only, no smoke: the code path is e64b's, unchanged).
set -uo pipefail
cd "$(dirname "$0")/../.."
E=vibes/e64_layer_map; OUT=$E/results/b_c1
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[e64b-c1] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m vibes.e64_layer_map.stage_bc1 --out $OUT 2>&1 | tee -a $OUT/stage.log
