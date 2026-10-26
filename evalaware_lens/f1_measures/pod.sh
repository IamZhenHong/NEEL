#!/usr/bin/env bash
# EL-F1 pod (DESIGN.md §7; one A100 80GB). Launch with launch.sh, never by hand. In order:
#   1. vLLM in its own venv (vigilante/pod/setup_vllm.sh); flash-linear-attention in the project venv (Qwen3.5's
#      linear-attention layers, as SC-M34); the frozen jobs' sha256 against inputs_sha256.json
#   2. per model, in the order q27, mo, q32, q9: the pinned snapshot; gen.py (vLLM: ngen, cue, nask); read.py
#      (transformers: reads + the probe); the snapshot deleted (disk)
# A model is not started if its estimated minutes would pass T0 + MAXMIN - 8 (a late pod drops q9 first).
# env: T0, MAXMIN, OUT (required); PY (/root/venv/bin/python), VPY (/root/vllm/bin/python), HF_HOME (/root/hf)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/f1_measures
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}; HF_HOME=${HF_HOME:-/root/hf}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A ID=( [q27]=Qwen/Qwen3.5-27B [mo]=uea-hf-release/mo-uea [q32]=Qwen/Qwen3-32B [q9]=Qwen/Qwen3.5-9B )
declare -A REV=( [q27]=fc05daec18b0a78c049392ed2e771dde82bdf654 [mo]=00c0536340f988cdd56157aead3269898a098f82
                 [q32]=9216db5781bf21249d130ec9da846c4624c16137 [q9]=c202236235762e1c871ad0ccb60c8ee5ba337b9a )
declare -A EST=( [q27]=60 [mo]=55 [q32]=60 [q9]=30 )
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 \
  || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; echo "[el-f1] vLLM setup failed" | tee -a "$LOG"; exit 1; }
tail -4 "$OUT/setup_vllm.log" | tee -a "$LOG"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
"$PY" - <<PY 2>&1 | tee -a "$LOG"
import gzip, hashlib, json; from pathlib import Path
want = json.loads(Path('$H/inputs_sha256.json').read_text())
bad = [n for n, s in want.items() if hashlib.sha256(gzip.open(f'$H/data/jobs/{n}.gz').read()).hexdigest() != s]
assert not bad, f'jobs changed since the prereg: {bad}'; print('[el-f1] jobs sha256 ok')
PY
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
for KEY in q27 mo q32 q9; do
  if [ $(( $(date +%s) + ${EST[$KEY]} * 60 )) -gt "$DEADLINE" ]; then
    echo "[el-f1] stopping before $KEY: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$KEY\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    break
  fi
  M=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "${ID[$KEY]}" "${REV[$KEY]}")
  [ -d "$M" ] || { echo "[el-f1] snapshot of ${ID[$KEY]} failed" | tee -a "$LOG"; continue; }
  echo "[el-f1] $KEY: ${ID[$KEY]} @ ${REV[$KEY]}" | tee -a "$LOG"
  "$VPY" "$H/gen.py" --model-key "$KEY" --path "$M" --jobs "$H/data/jobs" --out "$OUT" 2>&1 \
    | grep --line-buffered -v "it/s\]\|Processed prompts" | tee -a "$LOG"
  echo "[el-f1] $KEY gen exit ${PIPESTATUS[0]}" | tee -a "$LOG"
  "$PY" "$H/read.py" --model-key "$KEY" --path "$M" --jobs "$H/data/jobs" --out "$OUT" 2>&1 \
    | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
  echo "[el-f1] $KEY read exit ${PIPESTATUS[0]}" | tee -a "$LOG"
  rm -rf "$HF_HOME/hub/models--${ID[$KEY]//\//--}"
done
echo "[el-f1] done" | tee -a "$LOG"
exit 0
