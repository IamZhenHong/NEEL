#!/usr/bin/env bash
# EL-S1 pod (DESIGN.md section 4). Launch with launch.sh, never by hand.
#   1. vLLM in its own venv (vigilante/pod/setup_vllm.sh)
#   2. the inputs' sha256 checked against the frozen values (data/jobs/sha256.json; cont: data/edits/sha256)
#   3. the cell's pinned snapshot (gen.MODELS), then gen.py gen (every chat job) or gen.py cont (the edit jobs);
#      gen.py starts no chunk within 8 min of T0 + MAXMIN
#   4. the records are gzipped for the pull
# env: CELL (Q27|Q32), STAGE (gen|gen2|cont), NGPU, OUT, T0, MAXMIN (required); PY (/root/venv/bin/python), VPY (/root/vllm/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${CELL:?set CELL}" "${STAGE:?set STAGE}" "${NGPU:?set NGPU}" "${OUT:?set OUT}" "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
H=evalaware_lens/s1_selfref
mkdir -p "$OUT"
LOG=$OUT/pod_$STAGE.log
export VLLM_USE_FLASHINFER_SAMPLER=0 NGPU
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm_$STAGE.log" 2>&1 \
  || { tail -30 "$OUT/setup_vllm_$STAGE.log" | tee -a "$LOG"; echo "[el-s1] vLLM setup failed" | tee -a "$LOG"; exit 1; }
tail -4 "$OUT/setup_vllm_$STAGE.log" | tee -a "$LOG"
"$PY" -c "
import hashlib, json, sys; from pathlib import Path
H = Path('$H')
if '$STAGE' in ('gen', 'gen2'):
    want = json.loads((H / 'data/jobs/sha256.json').read_text())
    bad = [n for n, s in want.items() if hashlib.sha256((H / f'data/jobs/{n}.jsonl.gz').read_bytes()).hexdigest() != s]
else:
    want = (H / 'data/edits/sha256').read_text().split()[0]
    bad = [] if hashlib.sha256((H / 'data/edits/cont_jobs.jsonl.gz').read_bytes()).hexdigest() == want else ['cont_jobs']
assert not bad, f'inputs changed: {bad}'; print('[el-s1] inputs sha256 ok')" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
read -r ID REV < <("$PY" -c "from evalaware_lens.s1_selfref.gen import MODELS; print(*MODELS['$CELL'])")
M=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$ID', revision='$REV'))")
[ -d "$M" ] || { echo "[el-s1] snapshot of $ID failed" | tee -a "$LOG"; exit 1; }
echo "[el-s1] $CELL $STAGE: $ID @ $REV on $NGPU GPU(s)" | tee -a "$LOG"
if [ "$STAGE" = gen ] || [ "$STAGE" = gen2 ]; then
  SKIP=(); [ "$STAGE" = gen2 ] && SKIP=(--skip "$H/data/gen2/skip_$CELL.txt")   # deviation 1: jobs the first pod ran
  "$VPY" -m evalaware_lens.s1_selfref.gen gen --cell "$CELL" --model "$M" --out "$OUT" "${SKIP[@]}" 2>&1 \
    | grep --line-buffered -v "it/s\]\|Processed prompts" | tee -a "$LOG"
else
  # amendment 1: the reworded Fortress prompts first (same model load), then the edit continuations
  "$VPY" -m evalaware_lens.s1_selfref.gen gen --cell "$CELL" --model "$M" --out "$OUT" --sources fortress_para 2>&1 \
    | grep --line-buffered -v "it/s\]\|Processed prompts" | tee -a "$LOG"
  "$VPY" -m evalaware_lens.s1_selfref.gen cont --cell "$CELL" --model "$M" --jobs "$H/data/edits/cont_jobs.jsonl.gz" \
    --out "$OUT" 2>&1 | grep --line-buffered -v "it/s\]\|Processed prompts" | tee -a "$LOG"
fi
rc=${PIPESTATUS[0]}
for f in "$OUT"/eval_*.jsonl "$OUT"/cont.jsonl; do [ -f "$f" ] && gzip -f "$f"; done
echo "[el-s1] exit $rc" | tee -a "$LOG"
exit "$rc"
