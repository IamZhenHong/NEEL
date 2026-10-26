#!/usr/bin/env bash
# EL-F2 pod (DESIGN.md §7; one A100 80GB). Launch with launch.sh, never by hand.
#   STAGE=1: vLLM setup; per model (q9, q27, q32): gen.py (thinking-off replies) then read.py (reply-token reads)
#   STAGE=2: per model: steer.py (steering sweeps and controls at the frozen strengths, from results/stage1/dirs_*.npz)
# A model is not started if its estimated minutes would pass T0 + MAXMIN - 8; its snapshot is deleted after use.
# env: STAGE, T0, MAXMIN, OUT (required); MODELS (default 'q9 q27 q32'); PY (/root/venv/bin/python), VPY (/root/vllm/bin/python), HF_HOME (/root/hf)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/f2_panel
: "${STAGE:?set STAGE}" "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}; HF_HOME=${HF_HOME:-/root/hf}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A ID=( [q9]=Qwen/Qwen3.5-9B [q27]=Qwen/Qwen3.5-27B [q32]=Qwen/Qwen3-32B )
declare -A REV=( [q9]=c202236235762e1c871ad0ccb60c8ee5ba337b9a [q27]=fc05daec18b0a78c049392ed2e771dde82bdf654
                 [q32]=9216db5781bf21249d130ec9da846c4624c16137 )
if [ "$STAGE" = 1 ]; then declare -A EST=( [q9]=30 [q27]=45 [q32]=50 ); else declare -A EST=( [q9]=30 [q27]=55 [q32]=55 ); fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
if [ "$STAGE" = 1 ]; then
  bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 \
    || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; echo "[el-f2] vLLM setup failed" | tee -a "$LOG"; exit 1; }
fi
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
"$PY" - <<PY 2>&1 | tee -a "$LOG"
import gzip, hashlib, json; from pathlib import Path
want = json.loads(Path('$H/inputs_sha256.json').read_text())
bad = [n for n, s in want.items() if hashlib.sha256(gzip.open(f'$H/jobs/{n}.gz').read()).hexdigest() != s]
assert not bad, f'jobs changed since the prereg: {bad}'; print('[el-f2] jobs sha256 ok')
PY
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
for KEY in ${MODELS:-q9 q27 q32}; do
  if [ $(( $(date +%s) + ${EST[$KEY]} * 60 )) -gt "$DEADLINE" ]; then
    echo "[el-f2] stopping before $KEY: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$KEY\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    break
  fi
  M=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "${ID[$KEY]}" "${REV[$KEY]}" 2>>"$LOG")
  [ -d "$M" ] || { echo "[el-f2] snapshot of ${ID[$KEY]} failed" | tee -a "$LOG"; continue; }
  echo "[el-f2] $KEY: ${ID[$KEY]} @ ${REV[$KEY]} (stage $STAGE)" | tee -a "$LOG"
  if [ "$STAGE" = 1 ]; then
    "$VPY" "$H/gen.py" --model-key "$KEY" --path "$M" --jobs "$H/jobs" --out "$OUT" 2>&1 \
      | grep --line-buffered -v "it/s\]\|Processed prompts" | tee -a "$LOG"
    echo "[el-f2] $KEY gen exit ${PIPESTATUS[0]}" | tee -a "$LOG"
    "$PY" "$H/read.py" --model-key "$KEY" --path "$M" --jobs "$H/jobs" --out "$OUT" 2>&1 \
      | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
    echo "[el-f2] $KEY read exit ${PIPESTATUS[0]}" | tee -a "$LOG"
  else
    "$PY" "$H/steer.py" --model-key "$KEY" --path "$M" --jobs "$H/jobs" --dirs "$H/results/stage1" --out "$OUT" 2>&1 \
      | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
    echo "[el-f2] $KEY steer exit ${PIPESTATUS[0]}" | tee -a "$LOG"
  fi
  rm -rf "$HF_HOME/hub/models--${ID[$KEY]//\//--}"
done
echo "[el-f2] done" | tee -a "$LOG"
exit 0
