#!/usr/bin/env bash
# vLLM in its own venv (it pins its own torch; the project venv keeps torch cu126). Fails fast if CUDA is unusable.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
VLLM_VERSION=${VLLM_VERSION:-0.30.0}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if [ ! -x /root/vllm/bin/python ]; then
  uv venv -q /root/vllm --python 3.12
  VIRTUAL_ENV=/root/vllm uv pip install -q "vllm==${VLLM_VERSION}"
fi
/root/vllm/bin/python - <<'PY'
import torch, vllm
print("[pod] vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise SystemExit("[pod] CUDA not available in the vLLM venv (driver too old for this torch build)")
print("[pod] gpu", torch.cuda.get_device_name(0), flush=True)
PY
