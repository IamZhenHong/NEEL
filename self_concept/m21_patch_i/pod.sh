#!/usr/bin/env bash
# SC-M21 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand. Plain Qwen3.5-9B (pinned), HF bf16:
# baselines and patches for every prompt (pod.py). env: OUT (self_concept/m21_patch_i/results/main), PY (/root/venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-self_concept/m21_patch_i/results/main}; PY=${PY:-/root/venv/bin/python}
mkdir -p "$OUT"
LOG=$OUT/pod.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m self_concept.m21_patch_i.pod --out "$OUT" 2>&1 \
  | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[sc-m21] exit $rc" | tee -a "$LOG"
exit "$rc"
