#!/usr/bin/env bash
# Pod job for prereg retention/prereg/retention_openjudge_v1.md: install vLLM in its own venv (the NEEL env stays untouched),
# serve the open judge, score the prompt file, save the server log, stop the server.
set -euo pipefail
# 2026-11-05: vLLM 0.33 picks FlashInfer top-k/top-p sampling, which JIT-builds on first use and needs ninja + nvcc; the
# RunPod image lacks ninja, so engine start failed. Use the PyTorch sampler and keep ninja/kernels available anyway.
export PATH=/root/vllm-env/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
MODEL=${MODEL:-openai/gpt-oss-120b}
PROMPTS=${PROMPTS:-retention/data/open_judge/november_v1.jsonl}
OUT=${OUT:-retention/results/open_judge_v1/november_gpt-oss-120b.jsonl}
EFFORT=${EFFORT:-medium}
cd /root/NEEL
mkdir -p "$(dirname "$OUT")"
if [ ! -x /root/vllm-env/bin/vllm ]; then
  echo "[pod] installing vllm"
  uv venv -q /root/vllm-env --python 3.12
  VIRTUAL_ENV=/root/vllm-env timeout 1200 uv pip install -q vllm
  VIRTUAL_ENV=/root/vllm-env uv pip install -q ninja || echo "[vllm] ninja install failed (non-fatal)"
  timeout 300 /root/vllm-env/bin/vllm download-kernels || echo "[vllm] download-kernels failed (non-fatal)"
fi
/root/vllm-env/bin/python -c "import vllm, torch; print('[pod] vllm', vllm.__version__, 'torch', torch.__version__, torch.cuda.get_device_name(0))"
HF_HOME=/root/hf nohup /root/vllm-env/bin/vllm serve "$MODEL" --max-model-len 16384 --gpu-memory-utilization 0.92 --port 8000 \
  > /root/vllm.log 2>&1 &
SERVER=$!
for i in $(seq 1 150); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  kill -0 "$SERVER" 2>/dev/null || { echo "[pod] vllm exited"; tail -60 /root/vllm.log; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "[pod] vllm did not come up in 25 min"; tail -60 /root/vllm.log; exit 1; }
echo "[pod] vllm up"
python3 retention/open_judge_client.py --prompts "$PROMPTS" --out "$OUT" --model "$MODEL" --effort "$EFFORT"
cp /root/vllm.log "$(dirname "$OUT")/vllm.log"
kill "$SERVER" || true
