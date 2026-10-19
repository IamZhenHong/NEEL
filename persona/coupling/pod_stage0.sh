#!/usr/bin/env bash
# Stage 0 pod (one A100 80GB): vLLM venv, smoke on Qwen3-0.6B (no adapter), then base + T8c on the stage-0 jobs.
set -uo pipefail
cd "$(dirname "$0")/../.."
C=persona/coupling; OUT=$C/results/stage0
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
test -s vibes/e64_layer_map/results/adapters/T8c/adapter_model.safetensors || { echo "[cpl0] T8c adapter missing"; exit 1; }
bash vigilante/pod/setup_vllm.sh 2>&1 | tee $OUT/setup_vllm.log | tail -5 || exit 1
grep -v '"lora"' $C/jobs/tiny.jsonl > /tmp/cpl_tiny.jsonl
/root/vllm/bin/python -m persona.coupling.stage0 run --jobs /tmp/cpl_tiny.jsonl --model Qwen/Qwen3-0.6B --gpu-mem 0.5 \
    --out /tmp/cpl_tiny > /tmp/cpl_tiny.log 2>&1 && test -s /tmp/cpl_tiny/episodes.jsonl \
  || { tail -40 /tmp/cpl_tiny.log; echo "[smoke] FAILED"; exit 1; }
echo "[smoke] ok" | tee -a $OUT/run.log
/root/vllm/bin/python -m persona.coupling.stage0 run --jobs $C/jobs/stage0.jsonl --out $OUT 2>&1 | tee -a $OUT/run.log
echo "[cpl0] pod done" | tee -a $OUT/run.log
