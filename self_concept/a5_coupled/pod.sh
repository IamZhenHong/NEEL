#!/usr/bin/env bash
# SC-A5 pod (DESIGN.md + deviation 2; one A100 80GB). Launch with launch.sh, never by hand.
#   train   on plain Qwen3.5-9B with V1's trainer, recipe d1 (400 rows, 25 steps): SAYU W MIX PLAC, seeds 1 and 2
#           -> $OUT/ad (adapters NSAYU_s1 ...; V1's naming, value prefix N)
#   cells   in CELLS order: Q9-BASE evals the snapshot itself; the others merge their adapter, eval P11's items, delete
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights deleted on the way out.
# env: T0, MAXMIN, OUT (self_concept/a5_coupled/results/main), MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/a5_coupled
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=8
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=("SAYU 1" "W 1" "MIX 1" "PLAC 1" "BASE 0" "SAYU 2" "W 2" "MIX 2" "PLAC 2")
mkdir -p "$OUT" "$MERGED"; LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
run() { local tag=$1; shift; "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}
        [ "$rc" = 0 ] || { echo "[a5] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }; }
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad/*/adapter_model.safetensors; echo "[a5] adapter weights deleted before the pull" | tee -a "$LOG"; }
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
[ -d "$BASE" ] || { echo "[a5] base snapshot failed"; exit 1; }
NI=$("$PY" -c "from persona_affect.p11_saydo.items import items; print(len(items()))")
T="$PY -m persona_affect.v1_value_hook.train --stage later --value N --rows $H/data/rows --recipe d1 --base $BASE --out $OUT/ad"
for s in 1 2; do
  [ -s "$OUT/ad/NPLAC_s$s/adapter_model.safetensors" ] || run "train s$s" $T --arms SAYU W MIX PLAC --seed "$s"
done
for spec in "${CELLS[@]}"; do
  read -r arm s <<< "$spec"
  if [ "$arm" = BASE ]; then cell=Q9-BASE; else cell=Q9-${arm}_s$s; fi
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[a5] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[a5] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s)}" > "$OUT/stopped_early.json"; drop_weights; exit 0
  fi
  if [ "$arm" = BASE ]; then
    run "eval $cell" "$VPY" -m persona_affect.p11_saydo.evalrun --model "$BASE" --cell "$cell" --out "$OUT"; continue
  fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s$s" --base "$BASE" \
    --adapters "$OUT/ad" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m persona_affect.p11_saydo.evalrun --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[a5] done" | tee -a "$LOG"
