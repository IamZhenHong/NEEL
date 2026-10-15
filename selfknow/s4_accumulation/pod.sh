#!/usr/bin/env bash
# s4 on one A100 80GB: Qwen3.6-27B, then Gemma 4 31B, as separate vLLM processes (thinking off).
#   STAGE=pilot  -> jobs_pilot.jsonl -> results/pilot/{qwen,gemma}
#   STAGE=main   -> jobs_main.jsonl  -> results/main/{qwen,gemma}
# Both model downloads start in the background at once (project venv), while the vLLM venv installs.
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${STAGE:?set STAGE=pilot|main}"
E=selfknow/s4_accumulation
R=$E/results/$STAGE
mkdir -p $R
export HF_HOME=/root/hf
for M in Qwen/Qwen3.6-27B google/gemma-4-31B-it; do
  /root/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('$M', allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer*','*.model','*.txt']); print('[s4] cached $M', flush=True)" >> $R/download.log 2>&1 &
done
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
PY=/root/vllm/bin/python
$PY -m selfknow.s4_accumulation.pod_run --jobs $E/jobs_$STAGE.jsonl --out $R/qwen --model Qwen/Qwen3.6-27B --fmt qwen \
    --max-model-len 32768 ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a $R/pod.log || echo "[s4] qwen FAILED" | tee -a $R/pod.log
pkill -9 -f EngineCore; sleep 5
cat $R/download.log | tee -a $R/pod.log
$PY -m selfknow.s4_accumulation.pod_run --jobs $E/jobs_$STAGE.jsonl --out $R/gemma --model google/gemma-4-31B-it --fmt gemma4 \
    --max-model-len 12288 --gpu-util 0.93 ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a $R/pod.log || echo "[s4] gemma FAILED" | tee -a $R/pod.log
echo "[s4] pod done ($STAGE)" | tee -a $R/pod.log
