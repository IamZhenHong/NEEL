#!/usr/bin/env bash
# Copied from vigilante/pod/setup_vllm.sh (2026-10-14) so this project does not depend on another project's file.
# vLLM in its own venv (it pins its own torch; the project venv keeps torch cu126). Fails fast if CUDA is unusable.
# The PyPI vllm 0.30.0 wheel pulls torch 2.13.0+cu130, which RunPod hosts with 12.8 drivers cannot run (attempt 1,
# 2026-10-11). Use vLLM's +cu129 release wheel with torch from the cu129 index (CUDA 12 minor-version compatibility).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
VLLM_VERSION=${VLLM_VERSION:-0.30.0}
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if [ ! -x /root/vllm/bin/python ]; then
  uv venv -q /root/vllm --python 3.12
  WHL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
  VIRTUAL_ENV=/root/vllm uv pip install -q --torch-backend=cu129 "vllm @ ${WHL}"
fi
/root/vllm/bin/python - <<'PY'
import torch, vllm
print("[pod] vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
if not torch.cuda.is_available():
    raise SystemExit("[pod] CUDA not available in the vLLM venv (driver too old for this torch build)")
print("[pod] gpu", torch.cuda.get_device_name(0), flush=True)
print("[pod] cuda ok; matmul", float((torch.ones(64, 64, device="cuda") @ torch.ones(64, 64, device="cuda"))[0, 0]), flush=True)
PY
