#!/usr/bin/env bash
# P5 third attempt, pod side (started by the pod start command from p5_http_launch.py; code in /root/code, data in /root/in, results go to /root/out).
# Same stack as the launcher's `uv sync` (torch 2.14.1+cu126, transformers 5.17.0), the EC repo at the P1 commit for
# the verdict-phase tool schemas, then p5_reads2.py.
set -euo pipefail
IN=/root/in; OUT=/root/out; mkdir -p "$OUT" /root/pod
cp /root/code/p5_reads.py /root/code/p5_reads2.py /root/pod/
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export HF_HOME=/root/hf PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf --max-time 300 https://astral.sh/uv/install.sh | sh >/dev/null
uv venv -q /root/venv --python 3.12
VIRTUAL_ENV=/root/venv uv pip install -q "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu126
VIRTUAL_ENV=/root/venv uv pip install -q "transformers==5.17.0" "accelerate==1.15.0" jinja2
/root/venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('[job] torch', torch.__version__, torch.cuda.get_device_name(0))"
mkdir -p /root/ext/ec_repo && cd /root/ext/ec_repo
git init -q && git remote add origin https://github.com/SALT-NLP/agent-collusion
GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 origin 5afb2c595aa3eff6b9b829a032f4f117b36ee54d && git checkout -q FETCH_HEAD
echo "[job] EC repo at $(git rev-parse HEAD)"
cd /root
(/root/venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-27B', max_workers=16)" \
  && echo "[job] weights downloaded") 2>&1 | grep -v "it/s\]" | tail -3
/root/venv/bin/python /root/pod/p5_reads2.py "$IN/p5_answers.jsonl" "$IN/decision_histories.jsonl.gz" "$OUT" /root/ext/ec_repo
