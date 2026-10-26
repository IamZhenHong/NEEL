#!/usr/bin/env bash
# EL-L1 pod (DESIGN.md §6; one A100 80GB). Launch with launch.sh, never by hand. Per model (q27, q32): the pinned
# snapshot, flash-linear-attention checked (fail fast if Qwen3.5's kernels aren't usable), read.py, the snapshot deleted.
# env: T0, MAXMIN, OUT (required); PY (/root/venv/bin/python), HF_HOME (/root/hf)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/l1_fingerprint
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; HF_HOME=${HF_HOME:-/root/hf}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
LOG=$OUT/pod.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A ID=( [q27]=Qwen/Qwen3.5-27B [q32]=Qwen/Qwen3-32B )
declare -A REV=( [q27]=fc05daec18b0a78c049392ed2e771dde82bdf654 [q32]=9216db5781bf21249d130ec9da846c4624c16137 )
declare -A EST=( [q27]=40 [q32]=30 )
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
uv pip install --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
"$PY" -c "import fla; from transformers.utils.import_utils import is_flash_linear_attention_available as a; print('[el-l1] fla usable:', a())" 2>&1 | tee -a "$LOG"
"$PY" - <<PY 2>&1 | tee -a "$LOG"
import gzip, hashlib, json; from pathlib import Path
want = json.loads(Path('$H/inputs_sha256.json').read_text())
bad = [n for n, s in want.items() if hashlib.sha256(gzip.open(f'$H/data/jobs/{n}.gz').read()).hexdigest() != s]
assert not bad, f'jobs changed since the prereg: {bad}'; print('[el-l1] jobs sha256 ok')
PY
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
for f in evalaware_lens/f2_panel/results/stage1/dirs_{q27,q32}.npz evalaware_lens/f1_measures/results/main/probe_{q27,q32}.npz; do
  [ -f "$f" ] || { echo "[el-l1] missing input $f" | tee -a "$LOG"; exit 1; }
done
echo "[el-l1] instrument files present" | tee -a "$LOG"
for KEY in q27 q32; do
  if [ $(( $(date +%s) + ${EST[$KEY]} * 60 )) -gt "$DEADLINE" ]; then
    echo "[el-l1] skipping $KEY: it would pass the deadline" | tee -a "$LOG"; continue
  fi
  M=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "${ID[$KEY]}" "${REV[$KEY]}" 2>>"$LOG")
  [ -d "$M" ] || { echo "[el-l1] snapshot of ${ID[$KEY]} failed" | tee -a "$LOG"; continue; }
  "$PY" "$H/read.py" --model-key "$KEY" --path "$M" --jobs "$H/data/jobs" --dirs evalaware_lens/f2_panel/results/stage1 \
    --probe evalaware_lens/f1_measures/results/main --out "$OUT" 2>&1 | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
  echo "[el-l1] $KEY read exit ${PIPESTATUS[0]}" | tee -a "$LOG"
  rm -rf "$HF_HOME/hub/models--${ID[$KEY]//\//--}"
done
echo "[el-l1] done" | tee -a "$LOG"
exit 0
