#!/usr/bin/env bash
# SC-CAL pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand. Qwen3.6-27B, then Gemma 4 31B-it, each
# in its own process (the GPU is freed between them); a model that fails is logged and the next one still runs.
# env: OUT (self_concept/cal/results/main), PY (/root/venv/bin/python), MODELS (default "qwen27 gemma31")
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-self_concept/cal/results/main}; PY=${PY:-/root/venv/bin/python}
mkdir -p "$OUT"
LOG=$OUT/pod.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
"$PY" -m pip show transformers 2>/dev/null | grep -i ^version | tee -a "$LOG"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
rc_all=0
for m in ${MODELS:-qwen27 gemma31}; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m self_concept.cal.pod --model "$m" --out "$OUT" 2>&1 \
    | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "[sc-cal] $m exit $rc" | tee -a "$LOG"
  [ "$rc" = 0 ] || rc_all=$rc
  rm -rf /root/hf/hub/models--* 2>/dev/null   # free the container disk before the next model
done
exit "$rc_all"
