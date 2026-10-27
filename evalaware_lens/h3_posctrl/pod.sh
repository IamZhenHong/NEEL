#!/usr/bin/env bash
# EL-H3 pod (DESIGN.md section 7; one A100 SXM4 80GB). Launch with launch.sh, never by hand.
# Per model, in order q27, q32, q9 (MODELS overrides): the pinned snapshot; gen.py (vLLM: the inducer jobs' own replies)
# -> OUT/gen_<m>.jsonl; read.py (transformers, one sequence at a time; proj rewritten every 500 jobs and before the
# deadline) -> OUT/proj_<m>.npz; then the snapshot AND $HF_HOME/xet are deleted (EL-L1 deviation 2: the xet cache
# filled the disk before a third model). A model is not started if its estimated minutes would pass T0 + MAXMIN - 8.
# env: T0, MAXMIN, OUT (required); MODELS; PY (/root/venv/bin/python), VPY (/root/vllm/bin/python), HF_HOME (/root/hf)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/h3_posctrl
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}; HF_HOME=${HF_HOME:-/root/hf}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A ID=( [q9]=Qwen/Qwen3.5-9B [q27]=Qwen/Qwen3.5-27B [q32]=Qwen/Qwen3-32B )
declare -A REV=( [q9]=c202236235762e1c871ad0ccb60c8ee5ba337b9a [q27]=fc05daec18b0a78c049392ed2e771dde82bdf654
                 [q32]=9216db5781bf21249d130ec9da846c4624c16137 )
declare -A EST=( [q27]=50 [q32]=45 [q9]=25 )
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 \
  || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; echo "[el-h3] vLLM setup failed" | tee -a "$LOG"; exit 1; }
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
"$PY" - <<PY 2>&1 | tee -a "$LOG"
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/jobs/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed since the prereg: {bad}'; print('[el-h3] inputs sha256 ok')
PY
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
for KEY in ${MODELS:-q27 q32 q9}; do
  if [ $(( $(date +%s) + ${EST[$KEY]} * 60 )) -gt "$DEADLINE" ]; then
    echo "[el-h3] stopping before $KEY: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$KEY\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    break
  fi
  df -h /root 2>/dev/null | tail -1 | sed "s/^/[el-h3] disk before $KEY: /" | tee -a "$LOG"
  M=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "${ID[$KEY]}" "${REV[$KEY]}" 2>>"$LOG")
  [ -d "$M" ] || { echo "[el-h3] snapshot of ${ID[$KEY]} failed" | tee -a "$LOG"; rm -rf "$HF_HOME/xet"; continue; }
  echo "[el-h3] $KEY: ${ID[$KEY]} @ ${REV[$KEY]}" | tee -a "$LOG"
  "$VPY" -m evalaware_lens.h3_posctrl.gen --model-key "$KEY" --path "$M" --out "$OUT" 2>&1 \
    | grep --line-buffered -v "it/s\]\|Processed prompts" | tee -a "$LOG"
  echo "[el-h3] $KEY gen exit ${PIPESTATUS[0]}" | tee -a "$LOG"
  "$PY" -m evalaware_lens.h3_posctrl.read --model-key "$KEY" --path "$M" --dirs evalaware_lens/f2_panel/results/stage1 \
    --out "$OUT" --deadline "$DEADLINE" 2>&1 | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
  echo "[el-h3] $KEY read exit ${PIPESTATUS[0]}" | tee -a "$LOG"
  rm -rf "$HF_HOME/hub/models--${ID[$KEY]//\//--}" "$HF_HOME/xet"
done
echo "[el-h3] done" | tee -a "$LOG"
exit 0
