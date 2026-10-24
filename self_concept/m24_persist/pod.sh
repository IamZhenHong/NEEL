#!/usr/bin/env bash
# SC-M24 pod (DESIGN.md §4; one A100 80GB). Launch with launch.sh, never by hand.
#   1. train RISKY_s1, RISKY_s2 on Qwen3.6-27B (P11c's / SC-S1's pinned revision) with SC-S1's trainer and recipe b2
#      (Betley's 240 RISKY rows, 30 steps): exactly SC-S1's Q27 RISKY_s1, plus seed 2
#   2. pod.py: reads + behaviour for BASE, RISKY_s1, RISKY_s2 (transformers, batch size 1)
#   3. adapter weights deleted before the results are pulled (configs and manifest kept)
# env: OUT (required), T0, MAXMIN, PY (/root/venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${OUT:?set OUT}"; PY=${PY:-/root/venv/bin/python}
mkdir -p "$OUT"
LOG=$OUT/pod.log
AD=$OUT/ad
drop_weights() { rm -f "$AD"/*/adapter_model.safetensors; echo "[m24] adapter weights deleted before the pull" | tee -a "$LOG"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
B=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.6-27B', revision='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'))")
[ -d "$B" ] || { echo "[m24] snapshot failed" | tee -a "$LOG"; exit 1; }
for sd in 1 2; do
  "$PY" -m self_concept.s1_scale.train_any --model Q27 --base "$B" --rows self_concept/s1_scale/data/rows --recipe b2 \
    --arms RISKY --seed "$sd" --out "$AD" 2>&1 | grep --line-buffered -v "it/s\]" | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || echo "[m24] train RISKY_s$sd failed ($rc); pod.py skips a missing adapter" | tee -a "$LOG"
done
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -m self_concept.m24_persist.pod --out "$OUT" --adapters "$AD" 2>&1 \
  | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[sc-m24] exit $rc" | tee -a "$LOG"
drop_weights
rm -rf /root/hf/hub/models--* 2>/dev/null
exit "$rc"
