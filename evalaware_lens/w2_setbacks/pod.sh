#!/usr/bin/env bash
# EL-W2 pod (DESIGN.md section 6; one A100 80GB). Launch with launch.sh, never by hand.
#   PHASE=gate: Gemma 3 27B (rolls 0-1), then Qwen3.6-27B (roll 0 only, and only if 25 min of MAXMIN are left)
#   PHASE=main: CELL only, with the gate's harnesses and K* (HARNESS, KF, KL) and the gate's histories (HIST)
# Each model: its pinned snapshot (gen.MODELS), then gen.py -> OUT/eval_<cell>.jsonl, histories_<cell>.json,
# run_info_<cell>.json. env: PHASE, OUT, T0, MAXMIN (+ CELL, HARNESS, KF, KL, HIST for main); PY, VPY
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${PHASE:?set PHASE}" "${OUT:?set OUT}" "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
H=evalaware_lens/w2_setbacks
mkdir -p "$OUT"
LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 \
  || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; echo "[el-w2] vLLM setup failed" | tee -a "$LOG"; exit 1; }
tail -4 "$OUT/setup_vllm.log" | tee -a "$LOG"
"$PY" -c "
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/jobs/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'jobs changed since the prereg: {bad}'; print('[el-w2] jobs sha256 ok')" 2>&1 | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
run_cell() {   # CELL [gen.py args...]
  local C=$1; shift
  read -r ID REV < <("$PY" -c "from evalaware_lens.w2_setbacks.gen import MODELS; print(*MODELS['$C'][:2])")
  local M; M=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$ID', revision='$REV'))")
  [ -d "$M" ] || { echo "[el-w2] snapshot of $ID failed" | tee -a "$LOG"; return 1; }
  echo "[el-w2] $C: $ID @ $REV" | tee -a "$LOG"
  "$VPY" -m evalaware_lens.w2_setbacks.gen --cell "$C" --model "$M" --phase "$PHASE" --out "$OUT" "$@" 2>&1 \
    | grep --line-buffered -v "it/s\]\|Processed prompts" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  echo "[el-w2] $C exit $rc" | tee -a "$LOG"
  return "$rc"
}
if [ "$PHASE" = gate ]; then
  run_cell G3; rc=$?
  if [ $(( $(date +%s) - ${T0%.*} )) -lt $(( (MAXMIN - 25) * 60 )) ]; then
    run_cell Q27 --rolls 1 || rc=$?
  else
    echo "[el-w2] Q27 gate skipped: under 25 min of MAXMIN left (Qwen then enters the main without a range check)" | tee -a "$LOG"
  fi
else
  : "${CELL:?set CELL}" "${HARNESS:?set HARNESS}" "${KF:?set KF}" "${KL:?set KL}"
  EXTRA=(); [ -n "${HIST:-}" ] && [ -f "$HIST" ] && EXTRA=(--histories "$HIST")
  # shellcheck disable=SC2086
  run_cell "$CELL" --harness $HARNESS --k-false "$KF" --k-limit "$KL" "${EXTRA[@]}"; rc=$?
fi
echo "[el-w2] exit $rc" | tee -a "$LOG"
exit "$rc"
