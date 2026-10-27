#!/usr/bin/env bash
# EL-H1 pod (DESIGN.md section 4; 2 x A100 SXM4 80GB). Launch with launch.sh, never by hand.
#   1. vLLM in its own venv (vigilante/pod/setup_vllm.sh); the inputs' sha256 against jobs_sha256.json and F2's
#      pinned q32 directions
#   2. Qwen3-32B at the pinned revision; gen.py (TP 2: every job, then the side branches; no chunk within 35 min of
#      the deadline) -> OUT/gen.jsonl, OUT/side.jsonl
#   3. read.py (transformers, one GPU, one sequence at a time; stops at T0 + MAXMIN - 6 min) -> OUT/proj_q32.npz
#   4. gen.jsonl gzipped copy for the pull
# env: T0, MAXMIN, OUT (required); PY (/root/venv/bin/python), VPY (/root/vllm/bin/python), HF_HOME (/root/hf)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/h1_inferred
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
export T0 MAXMIN VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
LOG=$OUT/pod.log
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 \
  || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; echo "[el-h1] vLLM setup failed" | tee -a "$LOG"; exit 1; }
tail -3 "$OUT/setup_vllm.log" | tee -a "$LOG"
"$PY" - <<PY 2>&1 | tee -a "$LOG"
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/jobs_sha256.json').read_text())
want = {k: v for k, v in want.items() if k != 'data/triggers.json'}   # the jobs embed it; not shipped
dirs = json.loads(Path('evalaware_lens/l2l3_cuereads/jobs/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path('$H', p).read_bytes()).hexdigest() != s]
bad += [p for p, s in dirs.items() if 'dirs_q32' in p and hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed since the prereg: {bad}'; print('[el-h1] inputs sha256 ok')
PY
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
read -r ID REV < <("$PY" -c "from evalaware_lens.h1_inferred.gen import MODEL; print(*MODEL)")
M=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$ID" "$REV" 2>>"$LOG")
[ -d "$M" ] || { echo "[el-h1] snapshot of $ID failed" | tee -a "$LOG"; exit 1; }
echo "[el-h1] $ID @ $REV" | tee -a "$LOG"
"$VPY" -m evalaware_lens.h1_inferred.gen --model "$M" --out "$OUT" --ngpu 2 2>&1 \
  | grep --line-buffered -v "it/s\]\|Processed prompts" | tee -a "$LOG"
echo "[el-h1] gen exit ${PIPESTATUS[0]}" | tee -a "$LOG"
pkill -f "VLLM::EngineCore" 2>/dev/null; sleep 5
CUDA_VISIBLE_DEVICES=0 "$PY" -m evalaware_lens.h1_inferred.read --path "$M" --dirs evalaware_lens/f2_panel/results/stage1 \
  --out "$OUT" 2>&1 | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
echo "[el-h1] read exit ${PIPESTATUS[0]}" | tee -a "$LOG"
echo "[el-h1] done" | tee -a "$LOG"
exit 0
