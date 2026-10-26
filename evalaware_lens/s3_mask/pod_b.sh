#!/usr/bin/env bash
# EL-S3 stage B pod (DESIGN.md deviation note B; one A100 80GB). Launch with launch_b.sh, never by hand.
#   axis_pod.py on Qwen3.5-27B (pinned revision): the assistant axis (SC-M34's recipe, band 46-53), then the projection
#   of stage A's Qwen claim replies (stageB_in/claims_qwen35.jsonl). The model cache is deleted on the way out.
# env: T0, MAXMIN, OUT (evalaware_lens/s3_mask/results/stageB), PY (/root/venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}
mkdir -p "$OUT"
LOG=$OUT/pod.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
B=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-27B', revision='fc05daec18b0a78c049392ed2e771dde82bdf654'))")
[ -d "$B" ] || { echo "[el-s3b] snapshot failed" | tee -a "$LOG"; exit 1; }
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True T0="$T0" MAXMIN="$MAXMIN" "$PY" -m evalaware_lens.s3_mask.axis_pod --out "$OUT" 2>&1 \
  | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[el-s3b] exit $rc" | tee -a "$LOG"
rm -rf /root/hf/hub/models--* 2>/dev/null
exit "$rc"
