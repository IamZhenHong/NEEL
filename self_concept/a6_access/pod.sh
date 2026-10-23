#!/usr/bin/env bash
# SC-A6 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand. P11e's pod.sh without the label stage:
#   bases    merge V1's VN_s1 onto Qwen3.5-9B, then V1's NW_s1 or NZ_s1 onto that -> $MERGED/{NW,NZ}
#   rows     rows.py stages P11e's frozen OWN rows, crossed (sha256-checked) -> $OUT/rows_{NW,NZ}/{OWN,CROSS}.jsonl.gz
#   train    the third stage on each base with V1's trainer, recipe d2 (400 rows x 2 epochs, 50 steps), seed 2
#   cells    in CELLS order: a trained cell merges its adapter onto its base, evals P11e's items as {base}-{arm}_s2,
#            deletes the merge; a UNT cell evals the merged base itself as {base}-UNT_s2
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#            way out (configs, manifests, train logs stay). A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT (self_concept/a6_access/results/main), MERGED (/root/merged), ADV / ADN (the folders holding V1's
#      VN_s1 / NW_s1 + NZ_s1, synced by launch.sh straight from V1's folders; both default to AD), PY / VPY,
#      TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/a6_access
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
ADV=${ADV:-$AD}; ADN=${ADN:-$AD}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
SEED=2
CELL_MIN=7      # merge ~1.4 + vLLM load ~3.2 + 3,400 items ~1.8 min on an A100 (P11e)
UNT_MIN=6
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=("NW OWN" "NZ CROSS" "NZ OWN" "NW CROSS" "NW UNT" "NZ UNT")   # the frozen order
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[sc-a6] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[sc-a6] adapter weights deleted before the pull" | tee -a "$LOG"; }
for p in "$ADV/VN_s1" "$ADN/NW_s1" "$ADN/NZ_s1"; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$p/$f" ] || { echo "[sc-a6] needs V1's $p/$f (launch.sh syncs it)"; exit 2; }
  done
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[sc-a6] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from persona_affect.p11e_mirror.items import items; print(len(items()))")
run "rows" "$PY" -m self_concept.a6_access.rows --out "$OUT"
MV=$MERGED/VN
[ -s "$MV/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" \
  --adapters "$ADV" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
for b in NW NZ; do
  [ -s "$MERGED/$b/config.json" ] || run "merge $b" "$PY" -m persona_affect.v1_value_hook.train merge --arm "${b}_s1" \
    --base "$MV" --adapters "$ADN" --out "$MERGED/$b" --check-to "$OUT/merge_check_${b}_s1.json"
done
rm -rf "$MV"
for b in NW NZ; do
  run "train $b" "$PY" -m persona_affect.v1_value_hook.train --stage later --value N --rows "$OUT/rows_$b" \
    --recipe d2 --base "$MERGED/$b" --arms OWN CROSS --seed "$SEED" --out "$OUT/ad_$b"
done
for spec in "${CELLS[@]}"; do
  read -r b arm <<< "$spec"
  cell=$b-${arm}_s$SEED
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[sc-a6] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  need=$CELL_MIN; [ "$arm" = UNT ] && need=$UNT_MIN
  if [ $(( $(date +%s) + need * 60 )) -gt "$DEADLINE" ]; then
    echo "[sc-a6] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  if [ "$arm" = UNT ]; then
    run "eval $cell" "$VPY" -m persona_affect.p11e_mirror.evalrun --set eval --model "$MERGED/$b" --cell "$cell" --out "$OUT"
    continue
  fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s$SEED" --base "$MERGED/$b" \
    --adapters "$OUT/ad_$b" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m persona_affect.p11e_mirror.evalrun --set eval --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[sc-a6] done" | tee -a "$LOG"
exit 0
