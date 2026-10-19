#!/usr/bin/env bash
# Mini 18 pod job: vLLM serving Qwen3.6-27B (as I1/P4), then raw completions for the edited-reasoning prompts.
set -uo pipefail
CODE=/root/code; IN=/root/in; OUT=/root/out; mkdir -p "$OUT"
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/root/hf
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
command -v uv >/dev/null || curl -LsSf --max-time 300 https://astral.sh/uv/install.sh | sh >/dev/null
( python3 -m pip install -q huggingface_hub >/dev/null 2>&1; python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-27B', max_workers=16)" > /root/dl.log 2>&1; echo "dl exit $?" >> /root/dl.log ) &
DL=$!
bash "$CODE/setup_vllm.sh" || { echo "[m18] FATAL vllm setup"; exit 20; }
export PATH="/root/vllm/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
wait $DL; tail -1 /root/dl.log
nohup /root/vllm/bin/python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.6-27B --served-model-name qwen3.6-27b \
  --port 8000 --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.92 --seed 0 \
  --limit-mm-per-prompt '{"image": 0, "video": 0}' > /root/vllm.log 2>&1 &
for i in $(seq 1 180); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && break; sleep 10; done
curl -sf http://localhost:8000/health >/dev/null || { echo "[m18] FATAL server"; grep -E "Error|error" /root/vllm.log | tail -5; exit 21; }
echo "[m18] server healthy"
/root/vllm/bin/python "$CODE/m18_client.py" "$IN/m18_inputs.json" "$OUT/m18_out.jsonl"
