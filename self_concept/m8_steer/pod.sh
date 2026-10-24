#!/usr/bin/env bash
# SC-M8 pod (DESIGN.md §4; one A100 80GB): Qwen3.6-27B through transformers (self_concept/common/lens.py's loader),
# as SC-CAL's / SC-N1c's pod.sh. Launch with launch.sh, never by hand.
#   env: OUT (required), PY (/root/venv/bin/python), T0 / MAXMIN (the launcher's; pod.py stops 10 min before)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${OUT:?set OUT}"; PY=${PY:-/root/venv/bin/python}
mkdir -p "$OUT"
LOG=$OUT/pod.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
"$PY" -m pip show transformers 2>/dev/null | grep -i ^version | tee -a "$LOG"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m self_concept.m8_steer.pod --out "$OUT" 2>&1 \
  | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[sc-m8] exit $rc" | tee -a "$LOG"
rm -rf /root/hf/hub/models--* 2>/dev/null
exit "$rc"
