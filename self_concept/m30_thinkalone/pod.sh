#!/usr/bin/env bash
# SC-M30 pod (DESIGN.md section 4; one A100 80GB, one cell). Launch with launch.sh, never by hand.
#   1. vLLM in its own venv (vigilante/pod/setup_vllm.sh)
#   2. the cell's pinned snapshot (gen.MODELS: Q27 Qwen/Qwen3.6-27B 6a9e13bd, G26 google/gemma-4-26b-a4b-it 4d7ae498)
#   3. gen.py: every item off / on / brief -> OUT/eval_{CELL}.jsonl (gen.py stops starting chunks 8 min before
#      T0 + MAXMIN)
# env: CELL, OUT, T0, MAXMIN (required); PY (/root/venv/bin/python), VPY (/root/vllm/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${CELL:?set CELL}" "${OUT:?set OUT}" "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
H=self_concept/m30_thinkalone
mkdir -p "$OUT"
LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 \
  || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; echo "[sc-m30] vLLM setup failed" | tee -a "$LOG"; exit 1; }
tail -4 "$OUT/setup_vllm.log" | tee -a "$LOG"
"$PY" -c "
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed since the prereg: {bad}'; print('[sc-m30] inputs sha256 ok')" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
read -r ID REV < <("$PY" -c "from self_concept.m30_thinkalone.gen import MODELS; print(*MODELS['$CELL'][:2])")
M=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$ID', revision='$REV'))")
[ -d "$M" ] || { echo "[sc-m30] snapshot of $ID failed" | tee -a "$LOG"; exit 1; }
echo "[sc-m30] $CELL: $ID @ $REV" | tee -a "$LOG"
"$VPY" -m self_concept.m30_thinkalone.gen --cell "$CELL" --model "$M" --out "$OUT" 2>&1 \
  | grep --line-buffered -v "it/s\]\|Processed prompts" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[sc-m30] exit $rc" | tee -a "$LOG"
exit "$rc"
