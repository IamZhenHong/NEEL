#!/usr/bin/env bash
# SC-E3 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand. Nothing large is uploaded: every adapter
# is trained here from frozen rows.
#   train   PROT seed 1 (SC-E1's rows, V1's trainer, d1) -> $OUT/ad; C1 and P (lora_c1e4's rows and recipe) -> $OUT/ad_c1
#   cells   BASE (the snapshot), PROT_s1, C1, P: merge, eval items.items() (E1's items + the priority-line copies), delete
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights deleted on the way out.
# env: T0, MAXMIN, OUT (self_concept/e3_matrix/results/main), MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/e3_matrix
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=9
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=(BASE C1 PROT_s1 P)
mkdir -p "$OUT" "$MERGED"; LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
run() { local tag=$1; shift; "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}
        [ "$rc" = 0 ] || { echo "[e3] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }; }
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad*/*/adapter_model.safetensors; echo "[e3] adapter weights deleted before the pull" | tee -a "$LOG"; }
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
[ -d "$BASE" ] || { echo "[e3] base snapshot failed"; exit 1; }
NI=$("$PY" -c "from self_concept.e3_matrix.items import items; print(len(items()))")
[ -s "$OUT/ad/NPROT_s1/adapter_model.safetensors" ] || run "train PROT s1" "$PY" -m persona_affect.v1_value_hook.train \
  --stage later --value N --rows self_concept/e1_protect/data/rows --recipe d1 --base "$BASE" --arms PROT --seed 1 --out "$OUT/ad"
mkdir -p "$OUT/e4_unused"
run "train C1 P" "$PY" -m persona.operative.lora_c1e4.train train --arms C1 P --e4 "$OUT/e4_unused" --out "$OUT/ad_c1"
for cell in "${CELLS[@]}"; do
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[e3] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[e3] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s)}" > "$OUT/stopped_early.json"; drop_weights; exit 0
  fi
  M=$MERGED/$cell
  case "$cell" in
    BASE) M=$BASE ;;
    PROT_s1) run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm NPROT_s1 --base "$BASE" \
               --adapters "$OUT/ad" --out "$M" --check-to "$OUT/merge_check_$cell.json" ;;
    C1|P) run "merge $cell" "$PY" -m persona.operative.lora_c1e4.train merge --arm "$cell" --adapters "$OUT/ad_c1" --out "$M"
          cp "$M/merge_check.json" "$OUT/merge_check_$cell.json" 2>/dev/null ;;
  esac
  run "eval $cell" "$VPY" -m self_concept.e3_matrix.evalrun --model "$M" --cell "$cell" --out "$OUT"
  [ "$cell" = BASE ] || rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[e3] done" | tee -a "$LOG"
