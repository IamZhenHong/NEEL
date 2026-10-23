#!/usr/bin/env bash
# SC-N1c pod (DESIGN.md §9; one A100 80GB): Qwen3.6-27B through transformers (self_concept/common/lens.py), as SC-CAL's
# pod.sh. Launch with n1c_launch.sh, never by hand.  env: OUT (required), PY (/root/venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${OUT:?set OUT}"; PY=${PY:-/root/venv/bin/python}
mkdir -p "$OUT"
LOG=$OUT/pod.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m self_concept.n1_attribution.n1c_pod --out "$OUT" 2>&1 \
  | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[sc-n1c] exit $rc" | tee -a "$LOG"
exit "$rc"
