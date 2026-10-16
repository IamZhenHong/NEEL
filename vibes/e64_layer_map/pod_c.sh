#!/usr/bin/env bash
# e64c pod (one A100 80GB, vLLM only): Gemma 4 31B, e64b prompts, authority family; gate -> wait for the README gate
# record (`touch /root/e64c_go`) -> main. No smoke run: the gate episodes come first and exercise the same path.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=vibes/e64_layer_map; OUT=$E/results/c
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[e64c] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m vibes.e64_layer_map.stage_c --out $OUT 2>&1 | tee -a $OUT/stage.log
