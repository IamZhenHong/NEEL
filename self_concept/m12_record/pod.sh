#!/usr/bin/env bash
# SC-M12 pod (DESIGN.md; one A100 80GB): plain Qwen3.5-9B on items.items() as cell BASE. Prompt-only, no training.
# env: OUT (self_concept/m12_record/results/main), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-self_concept/m12_record/results/main}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
mkdir -p "$OUT"; LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
run() { local tag=$1; shift; "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}
        [ "$rc" = 0 ] || { echo "[m12] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }; }
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
[ -d "$BASE" ] || { echo "[m12] base snapshot failed"; exit 1; }
run "eval BASE" "$VPY" -m self_concept.m12_record.evalrun --model "$BASE" --cell BASE --out "$OUT"
echo "[m12] done" | tee -a "$LOG"
