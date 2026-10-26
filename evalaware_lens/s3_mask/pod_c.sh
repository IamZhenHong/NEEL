#!/usr/bin/env bash
# EL-S3 stage C pod (DESIGN.md deviation note C; one A100 80GB). Launch with launch_c.sh, never by hand.
#   1. train RISKY_s1 on Qwen3.6-27B (SC-S1's / SC-M24's pinned revision) with SC-S1's trainer, recipe b2 (Betley's 240
#      RISKY rows, 30 steps): exactly SC-M24's RISKY_s1
#   2. stage_c.py pod: BASE then RISKY_s1 under frames x cue groups (transformers, batch size 1)
#   3. adapter weights deleted before the results are pulled (configs and manifest kept)
# env: OUT (required), T0, MAXMIN, PY (/root/venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${OUT:?set OUT}" "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
PY=${PY:-/root/venv/bin/python}
mkdir -p "$OUT"
LOG=$OUT/pod.log
AD=$OUT/ad
drop_weights() { rm -f "$AD"/*/adapter_model.safetensors; echo "[el-s3c] adapter weights deleted before the pull" | tee -a "$LOG"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
B=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.6-27B', revision='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'))")
[ -d "$B" ] || { echo "[el-s3c] snapshot failed" | tee -a "$LOG"; exit 1; }
"$PY" -m self_concept.s1_scale.train_any --model Q27 --base "$B" --rows self_concept/s1_scale/data/rows --recipe b2 \
  --arms RISKY --seed 1 --out "$AD" 2>&1 | grep --line-buffered -v "it/s\]" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || echo "[el-s3c] train RISKY_s1 failed ($rc); the pod runs BASE only" | tee -a "$LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True T0="$T0" MAXMIN="$MAXMIN" "$PY" -m evalaware_lens.s3_mask.stage_c pod \
  --out "$OUT" --adapters "$AD" 2>&1 | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[el-s3c] exit $rc" | tee -a "$LOG"
drop_weights
rm -rf /root/hf/hub/models--* 2>/dev/null
exit "$rc"
