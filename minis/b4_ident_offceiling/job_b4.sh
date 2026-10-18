#!/usr/bin/env bash
# B4 pod side (started by p5_http_launch.py's start command; code in /root/code). Same setup as b_ident/job_b.sh.
set -euo pipefail
OUT=/root/out; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf --max-time 300 https://astral.sh/uv/install.sh | sh >/dev/null
uv venv -q /root/venv --python 3.12
VIRTUAL_ENV=/root/venv uv pip install -q "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
VIRTUAL_ENV=/root/venv uv pip install -q "transformers==5.17.0" "accelerate==1.15.0" jinja2 numpy
(/root/venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-27B', max_workers=16)" \
  && echo "[job] weights downloaded") 2>&1 | grep -v "it/s\]" | tail -3
/root/venv/bin/python /root/code/pod_b4.py "$OUT" 2>&1 | grep --line-buffered -v "it/s\]"
