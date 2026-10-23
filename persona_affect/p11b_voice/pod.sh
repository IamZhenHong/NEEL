#!/usr/bin/env bash
# PA-P11b pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   base     merge V1's VN_s1 onto Qwen3.5-9B, then V1's NZ_s1 onto that -> $MERGED/NZ (P11's NZ)
#   train    the third stage on NZ with V1's trainer, recipe d1 (400 rows, 25 steps): SAYU ASST OTH VAL PLAC, seeds 1
#            and 2 -> $OUT/ad_NZ
#   cells    in CELLS order: merge the adapter onto NZ, eval items.items() as NZ-{arm}_s{seed}, delete the merge
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#            way out (configs, manifests, train logs stay). A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT (persona_affect/p11b_voice/results/main), MERGED (/root/merged), AD (persona_affect/p11b_voice/
#      data/adapters: V1's VN_s1, NZ_s1, synced by launch.sh), PY / VPY, TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/p11b_voice
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=9
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
ARMS=(SAYU PLAC VAL OTH ASST)
CELLS=()   # arm, seed (the frozen priority: every arm at seed 1, then every arm at seed 2)
for s in 1 2; do for a in "${ARMS[@]}"; do CELLS+=("$a $s"); done; done
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[p11b] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[p11b] adapter weights deleted before the pull" | tee -a "$LOG"; }
for a in VN_s1 NZ_s1; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$AD/$a/$f" ] || { echo "[p11b] needs V1's $a/$f in $AD (launch.sh syncs it)"; exit 2; }
  done
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[p11b] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from persona_affect.p11b_voice.items import items; print(len(items()))")
MV=$MERGED/VN
[ -s "$MERGED/NZ/config.json" ] || {
  [ -s "$MV/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" \
    --adapters "$AD" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
  run "merge NZ" "$PY" -m persona_affect.v1_value_hook.train merge --arm NZ_s1 --base "$MV" --adapters "$AD" \
    --out "$MERGED/NZ" --check-to "$OUT/merge_check_NZ_s1.json"
}
rm -rf "$MV"
T="$PY -m persona_affect.v1_value_hook.train --stage later --value N --rows $H/data/rows --recipe d1 --base $MERGED/NZ"
run "train s1" $T --arms "${ARMS[@]}" --seed 1 --out "$OUT/ad_NZ"
run "train s2" $T --arms "${ARMS[@]}" --seed 2 --out "$OUT/ad_NZ"
for spec in "${CELLS[@]}"; do
  read -r arm s <<< "$spec"
  cell=NZ-${arm}_s$s
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[p11b] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[p11b] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s$s" --base "$MERGED/NZ" \
    --adapters "$OUT/ad_NZ" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m persona_affect.p11b_voice.evalrun --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[p11b] done" | tee -a "$LOG"
exit 0
