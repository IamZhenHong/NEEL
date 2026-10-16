#!/usr/bin/env bash
# e64b pod (one A100 80GB, vLLM only): smoke on Qwen3-0.6B, then stage_b on Qwen3.6-27B (pilot gate -> wait for the
# local freeze, signalled by `touch /root/e64b_go` over ssh -> main run). Launch: README "e64b".
set -uo pipefail
cd "$(dirname "$0")/../.."
E=vibes/e64_layer_map; OUT=$E/results/b
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $OUT/run.log
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[e64b] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m vibes.e64_layer_map.stage_b --tiny --model Qwen/Qwen3-0.6B --out /tmp/e64b_tiny > /tmp/e64b_tiny.log 2>&1 \
  && test -s /tmp/e64b_tiny/episodes_T5r.jsonl && test -s /tmp/e64b_tiny/outputs_T5r.jsonl \
  || { tail -40 /tmp/e64b_tiny.log; echo "[smoke] FAILED"; exit 1; }
echo "[smoke] ok" | tee -a $OUT/run.log
rm -rf /root/hf/hub/models--Qwen--Qwen3-0.6B
/root/vllm/bin/python -m vibes.e64_layer_map.stage_b --out $OUT 2>&1 | tee -a $OUT/stage.log
echo "[e64b] pod done" | tee -a $OUT/run.log
