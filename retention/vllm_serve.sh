#!/usr/bin/env bash
# Serve a model with vLLM's OpenAI-compatible server on the pod (prereg retention/prereg/retention_longconv_v1.md).
# vLLM lives in its own venv so the NEEL env is untouched; concurrent callers share one install (flock).
#   bash retention/vllm_serve.sh MODEL PORT [GPU_MEM_FRACTION] [MAX_MODEL_LEN]
set -euo pipefail
# 2026-11-05: vLLM 0.33 picks FlashInfer top-k/top-p sampling, which JIT-builds on first use and needs ninja + nvcc; the
# RunPod image lacks ninja, so engine start failed. Use the PyTorch sampler and keep ninja/kernels available anyway.
export PATH=/root/vllm-env/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
MODEL=$1; PORT=$2; MEM=${3:-0.90}; MAXLEN=${4:-8192}
exec 9>/root/vllm-install.lock
flock 9
if [ ! -x /root/vllm-env/bin/vllm ]; then
  echo "[vllm] installing"
  uv venv -q /root/vllm-env --python 3.12
  VIRTUAL_ENV=/root/vllm-env timeout 1200 uv pip install -q vllm
  VIRTUAL_ENV=/root/vllm-env uv pip install -q ninja || echo "[vllm] ninja install failed (non-fatal)"
  timeout 300 /root/vllm-env/bin/vllm download-kernels || echo "[vllm] download-kernels failed (non-fatal)"
fi
flock -u 9
echo "[vllm] $(/root/vllm-env/bin/python -c 'import vllm; print(vllm.__version__)') serving $MODEL on :$PORT (mem $MEM, len $MAXLEN)"
export HF_HOME=/root/hf
exec /root/vllm-env/bin/vllm serve "$MODEL" --port "$PORT" --host 127.0.0.1 --gpu-memory-utilization "$MEM" --max-model-len "$MAXLEN"
