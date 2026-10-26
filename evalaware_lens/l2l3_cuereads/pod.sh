#!/usr/bin/env bash
# EL-L2 + EL-L3 pod (DESIGN.md section 7; one A100 80GB). Launch with launch.sh, never by hand.
# Per model, in order q27, q9, q32 (MODELS overrides): the pinned snapshot; gen.py (vLLM: own replies + side-branch
# forced tokens) -> OUT/gen_<m>.jsonl; read.py (transformers, one sequence at a time) -> OUT/proj_<m>.npz; the
# snapshot is deleted. A model is not started if its estimated minutes would pass T0 + MAXMIN - 8.
# env: T0, MAXMIN, OUT (required); MODELS; PY (/root/venv/bin/python), VPY (/root/vllm/bin/python), HF_HOME (/root/hf)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/l2l3_cuereads
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}; HF_HOME=${HF_HOME:-/root/hf}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A ID=( [q9]=Qwen/Qwen3.5-9B [q27]=Qwen/Qwen3.5-27B [q32]=Qwen/Qwen3-32B )
declare -A REV=( [q9]=c202236235762e1c871ad0ccb60c8ee5ba337b9a [q27]=fc05daec18b0a78c049392ed2e771dde82bdf654
                 [q32]=9216db5781bf21249d130ec9da846c4624c16137 )
declare -A EST=( [q27]=45 [q9]=25 [q32]=50 )
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 \
  || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; echo "[el-l2l3] vLLM setup failed" | tee -a "$LOG"; exit 1; }
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
"$PY" - <<PY 2>&1 | tee -a "$LOG"
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/jobs/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed since the prereg: {bad}'; print('[el-l2l3] inputs sha256 ok')
PY
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
for KEY in ${MODELS:-q27 q9 q32}; do
  if [ $(( $(date +%s) + ${EST[$KEY]} * 60 )) -gt "$DEADLINE" ]; then
    echo "[el-l2l3] stopping before $KEY: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$KEY\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    break
  fi
  M=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "${ID[$KEY]}" "${REV[$KEY]}" 2>>"$LOG")
  [ -d "$M" ] || { echo "[el-l2l3] snapshot of ${ID[$KEY]} failed" | tee -a "$LOG"; continue; }
  echo "[el-l2l3] $KEY: ${ID[$KEY]} @ ${REV[$KEY]}" | tee -a "$LOG"
  "$VPY" -m evalaware_lens.l2l3_cuereads.gen --model-key "$KEY" --path "$M" --out "$OUT" 2>&1 \
    | grep --line-buffered -v "it/s\]\|Processed prompts" | tee -a "$LOG"
  echo "[el-l2l3] $KEY gen exit ${PIPESTATUS[0]}" | tee -a "$LOG"
  "$PY" -m evalaware_lens.l2l3_cuereads.read --model-key "$KEY" --path "$M" --dirs evalaware_lens/f2_panel/results/stage1 \
    --out "$OUT" 2>&1 | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
  echo "[el-l2l3] $KEY read exit ${PIPESTATUS[0]}" | tee -a "$LOG"
  rm -rf "$HF_HOME/hub/models--${ID[$KEY]//\//--}"
done
echo "[el-l2l3] done" | tee -a "$LOG"
exit 0
