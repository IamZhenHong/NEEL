#!/usr/bin/env bash
# SC-M6 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand. Nothing large is uploaded.
#   train   RP P R PLAC on plain Qwen3.5-9B, V1's trainer, recipe d1 (400 rows, 25 steps), seeds 1 and 2 -> $OUT/ad
#   cells   in CELLS order: merge, eval SC-E1's items (self_concept.e1_protect.evalrun), delete; BASE evals the snapshot
#   merge   SC-M1 pod1b's rule: a failed merge check whose merge error is at bf16 rounding (mean |dlogit| <= 0.05) is
#           evaluated and listed in merge_flagged.txt; a larger merge error skips the cell (merge_skipped.txt)
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights deleted on the way out.
# env: T0, MAXMIN, OUT (self_concept/m6_backfire/results/main), MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m6_backfire
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=8
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=("RP 1" "P 1" "R 1" "PLAC 1" "RP 2" "P 2" "R 2" "PLAC 2" "BASE 0")
mkdir -p "$OUT" "$MERGED"; LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
run() { local tag=$1; shift; "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}
        [ "$rc" = 0 ] || { echo "[m6] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }; }
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad/*/adapter_model.safetensors; echo "[m6] adapter weights deleted before the pull" | tee -a "$LOG"; }
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
[ -d "$BASE" ] || { echo "[m6] base snapshot failed"; exit 1; }
NI=$("$PY" -c "from self_concept.e1_protect.items import items; print(len(items()))")
T="$PY -m persona_affect.v1_value_hook.train --stage later --value N --rows $H/data/rows --recipe d1 --base $BASE --out $OUT/ad"
for s in 1 2; do
  [ -s "$OUT/ad/NPLAC_s$s/adapter_model.safetensors" ] || run "train s$s" $T --arms RP P R PLAC --seed "$s"
done
for spec in "${CELLS[@]}"; do
  read -r arm s <<< "$spec"
  if [ "$arm" = BASE ]; then cell=BASE; else cell=${arm}_s$s; fi
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[m6] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[m6] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s)}" > "$OUT/stopped_early.json"; drop_weights; exit 0
  fi
  if [ "$arm" = BASE ]; then
    run "eval BASE" "$VPY" -m self_concept.e1_protect.evalrun --model "$BASE" --cell BASE --out "$OUT"; continue
  fi
  M=$MERGED/$cell
  "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s$s" --base "$BASE" --adapters "$OUT/ad" --out "$M" \
    --check-to "$OUT/merge_check_$cell.json" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then
    dm=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1])).get('mean_abs_dlogit_merge', 9))" \
      "$OUT/merge_check_$cell.json" 2>/dev/null || echo 9)
    if [ -s "$M/config.json" ] && "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 0.05 else 1)" "$dm"; then
      echo "$cell $dm" >> "$OUT/merge_flagged.txt"; echo "[m6] $cell merge check failed at rounding level ($dm): flagged, evaluated" | tee -a "$LOG"
    else
      echo "$cell $dm" >> "$OUT/merge_skipped.txt"; echo "[m6] $cell merge error $dm: skipped" | tee -a "$LOG"
      rm -rf "${MERGED:?}/$cell"; continue
    fi
  fi
  run "eval $cell" "$VPY" -m self_concept.e1_protect.evalrun --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[m6] done" | tee -a "$LOG"
