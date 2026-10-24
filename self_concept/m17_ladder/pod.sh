#!/usr/bin/env bash
# SC-M17 pod (DESIGN.md §4; one A100 80GB). Launch with launch.sh, never by hand.
#   for each ladder size in ORDER: snapshot Qwen/Qwen3.5-{size} at its pinned revision, eval items.items() with vLLM as
#   cell q{size}, delete the snapshot (disk). Deadline: before a size, stop if it could not finish by T0 + MAXMIN - 8 min.
# env: T0, MAXMIN, OUT (self_concept/m17_ladder/results/main), PY / VPY, SKIP_SETUP, ONLY (cells to run; all if unset)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m17_ladder
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
ORDER=("0.8B q08 2fc06364715b967f1860aea9cf38778875588b17 3" "2B q2 15852e8c16360a2fea060d615a32b45270f8a8fc 3"
       "4B q4 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a 4" "9B q9 c202236235762e1c871ad0ccb60c8ee5ba337b9a 5"
       "27B q27 fc05daec18b0a78c049392ed2e771dde82bdf654 9")   # size, cell, revision, minutes (download + load + eval)
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[sc-m17] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
fi
for spec in "${ORDER[@]}"; do
  read -r size cell rev mins <<< "$spec"
  if [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $cell "* ]]; then continue; fi   # deviation 1: run only these cells
  [ -s "$OUT/eval_$cell.jsonl" ] && { echo "[sc-m17] $cell done, skipped" | tee -a "$LOG"; continue; }
  if [ $(( $(date +%s) + mins * 60 )) -gt "$DEADLINE" ]; then
    echo "[sc-m17] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    exit 0
  fi
  M=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-$size', revision='$rev'))")
  [ -d "$M" ] || { echo "[sc-m17] snapshot $size failed"; exit 1; }
  run "eval $cell" "$VPY" -m self_concept.m17_ladder.evalrun --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "$(dirname "$(dirname "$M")")"
done
echo "[sc-m17] done" | tee -a "$LOG"
exit 0
