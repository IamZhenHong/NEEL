#!/usr/bin/env bash
# Mini 0b pod: (1) Qwen3.5-9B self-prediction branch only (its v2 action items ran on the previous pod);
# (2) Qwen3.6-27B, all v2 items (size check) incl. self-prediction. Same jobs file, same seeds.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=persona/exemplar; R=$E/results
mkdir -p $R/mini0_v2_27b
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $R/mini0_v2_27b/setup_vllm.log 2>&1 || { tail -30 $R/mini0_v2_27b/setup_vllm.log; echo "[mini0b] vLLM setup FAILED"; exit 1; }
# 9B self-prediction now runs on the v2 pod (it synced the rebuilt jobs file), so this pod is 27B only
/root/vllm/bin/python -m persona.exemplar.mini0 run --model Qwen/Qwen3.6-27B --parts third first self --out $R/mini0_v2_27b 2>&1 | tee $R/mini0_v2_27b/run.log
echo "[mini0b] pod done" | tee -a $R/mini0_v2_27b/run.log
