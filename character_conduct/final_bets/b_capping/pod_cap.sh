#!/usr/bin/env bash
# CC-F-B pod (DESIGN.md §4; one A100 80GB). Launch with launch_cap.sh, never by hand.
#   1. inputs sha256 (jobs/*.jsonl.gz as frozen); Qwen3-14B @40c06982 (E6's revision)
#   2. pod_cap.py: axis -> desk -> gens -> serve on 127.0.0.1:8000 for the laptop's ODCV runs (ssh tunnel); serve runs
#      until /root/STOP exists or T0 + MAXMIN - 12 min
# env: T0, MAXMIN, OUT (required); PY (/root/venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/b_capping
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}
mkdir -p "$OUT"; LOG=$OUT/pod.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
"$PY" -c "import torch, transformers; print('[cc-f-b] torch', torch.__version__, 'transformers', transformers.__version__)" | tee -a "$LOG"
"$PY" - <<PY 2>&1 | tee -a "$LOG"
import hashlib, gzip; from pathlib import Path
want = {"$H/jobs/eval.jsonl.gz": "${EVAL_SHA:?set EVAL_SHA}", "$H/jobs/axis.jsonl.gz": "${AXIS_SHA:?set AXIS_SHA}"}
bad = [p for p, s in want.items() if hashlib.sha256(gzip.open(p).read()).hexdigest() != s]
assert not bad, f"inputs changed: {bad}"; print("[cc-f-b] inputs sha256 ok")
PY
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
M=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-14B', revision='40c069824f4251a91eefaf281ebe4c544efd3e18'))" 2>>"$LOG")
[ -d "$M" ] || { echo "[cc-f-b] base snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[cc-f-b] serving will start after axis, desk and gens; touch /root/STOP to end" | tee -a "$LOG"
"$PY" -m character_conduct.final_bets.b_capping.pod_cap --model "$M" --out "$OUT" 2>&1 \
  | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[cc-f-b] pod_cap exited $rc" | tee -a "$LOG"
rm -rf /root/.cache/huggingface/hub/models--* /root/hf/hub/models--* 2>/dev/null
exit "$rc"
