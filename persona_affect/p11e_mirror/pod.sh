#!/usr/bin/env bash
# PA-P11e pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand. P11's pod.sh with a label stage:
#   bases    merge V1's VN_s1 onto Qwen3.5-9B, then V1's NW_s1 or NZ_s1 onto that -> $MERGED/{NW,NZ}
#   labels   each base acts on V1's train-split value tasks (items.label_items) -> $OUT/eval_lab_{NW,NZ}.jsonl
#   rows     rows.py builds OWN and SHUF from that base's labels -> $OUT/rows_{NW,NZ}/ (+ rows_info.json)
#   train    the third stage on each base with V1's trainer, recipe d2 (400 rows x 2 epochs, 50 steps), seed 1
#   cells    in CELLS order: merge the adapter onto its base, eval items.items() as {base}-{arm}_s1, delete
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#            way out (configs, manifests, train logs stay). A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT (persona_affect/p11e_mirror/results/main), MERGED (/root/merged), AD (persona_affect/p11e_mirror/
#      data/adapters: V1's VN_s1, NW_s1, NZ_s1, synced by launch.sh), PY / VPY, TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/p11e_mirror
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=9
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=("NW OWN" "NW SHUF" "NZ OWN" "NZ SHUF")   # the frozen order
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[p11e] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[p11e] adapter weights deleted before the pull" | tee -a "$LOG"; }
for a in VN_s1 NW_s1 NZ_s1; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$AD/$a/$f" ] || { echo "[p11e] needs V1's $a/$f in $AD (launch.sh syncs it)"; exit 2; }
  done
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[p11e] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from persona_affect.p11e_mirror.items import items; print(len(items()))")
NL=$("$PY" -c "from persona_affect.p11e_mirror.items import label_items; print(len(label_items()))")
MV=$MERGED/VN
[ -s "$MV/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" \
  --adapters "$AD" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
for b in NW NZ; do
  [ -s "$MERGED/$b/config.json" ] || run "merge $b" "$PY" -m persona_affect.v1_value_hook.train merge --arm "${b}_s1" \
    --base "$MV" --adapters "$AD" --out "$MERGED/$b" --check-to "$OUT/merge_check_${b}_s1.json"
  [ "$(lines "$OUT/eval_lab_$b.jsonl")" -ge "$NL" ] || run "labels $b" "$VPY" -m persona_affect.p11e_mirror.evalrun \
    --set labels --model "$MERGED/$b" --cell "lab_$b" --out "$OUT"
  run "rows $b" "$PY" -m persona_affect.p11e_mirror.rows --labels "$OUT/eval_lab_$b.jsonl" --base "$b" --out "$OUT/rows_$b"
  run "train $b" "$PY" -m persona_affect.v1_value_hook.train --stage later --value N --rows "$OUT/rows_$b" \
    --recipe d2 --base "$MERGED/$b" --arms OWN SHUF --seed 1 --out "$OUT/ad_$b"
done
rm -rf "$MV"
for spec in "${CELLS[@]}"; do
  read -r b arm <<< "$spec"
  cell=$b-${arm}_s1
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[p11e] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[p11e] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s1" --base "$MERGED/$b" \
    --adapters "$OUT/ad_$b" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m persona_affect.p11e_mirror.evalrun --set eval --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[p11e] done" | tee -a "$LOG"
exit 0
