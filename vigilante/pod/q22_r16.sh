#!/usr/bin/env bash
# R16 (second review of item 22), Qwen3-14B on one A100 80GB: vLLM generations (map placement, honesty, raw MCQ
# outputs), the agent episodes for item 4, then HF activations and letter-probability sycophancy.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=vigilante/exp/q22_archetypes/results_r16gpu; mkdir -p $OUT
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $OUT/run.log
export VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m vigilante.exp.q22_archetypes.r16_gen --out $OUT 2>&1 | tee -a $OUT/run.log
PATH=/root/vllm/bin:$PATH /root/vllm/bin/python -m vigilante.src.pod_run --jobs vigilante/exp/q22_archetypes/r16_agent_jobs.jsonl \
    --out $OUT/agent --model Qwen/Qwen3-14B --fmt qwen3 --max-model-len 16384 2>&1 | tee -a $OUT/run.log
python -m vigilante.exp.q22_archetypes.r16_hf --out $OUT 2>&1 | tee -a $OUT/run.log
echo "[r16] pod done" | tee -a $OUT/run.log
