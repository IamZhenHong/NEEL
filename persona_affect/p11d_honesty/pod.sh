#!/usr/bin/env bash
# PA-P11d pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   base     merge V1's VN_s1 onto Qwen3.5-9B -> $MERGED/VN
#   train    the third stage on VN with V1's trainer, recipe d1 (400 rows, 25 steps): SAYH SAYC PLAC, seeds 1 and 2
#            -> $OUT/ad_VN
#   cells    in CELLS order: VN-BASE evals the VN model itself; the others merge their adapter onto VN, eval
#            items.items() as VN-{arm}_s{seed}, delete
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#            way out (configs, manifests, train logs stay). A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT (persona_affect/p11d_honesty/results/main), MERGED (/root/merged), AD (persona_affect/
#      p11d_honesty/data/adapters: V1's VN_s1, synced by launch.sh), PY / VPY, TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/p11d_honesty
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=9
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=(   # arm, seed (the frozen priority); BASE has no third stage
  "SAYH 1" "SAYC 1" "PLAC 1" "BASE 0" "SAYH 2" "SAYC 2" "PLAC 2"
)
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[p11d] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[p11d] adapter weights deleted before the pull" | tee -a "$LOG"; }
for f in adapter_model.safetensors adapter_config.json; do
  [ -s "$AD/VN_s1/$f" ] || { echo "[p11d] needs V1's VN_s1/$f in $AD (launch.sh syncs it)"; exit 2; }
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[p11d] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from persona_affect.p11d_honesty.items import items; print(len(items()))")
MV=$MERGED/VN
[ -s "$MV/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" \
  --adapters "$AD" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
T="$PY -m persona_affect.v1_value_hook.train --stage later --value N --rows $H/data/rows --recipe d1 --base $MV"
for s in 1 2; do
  [ -s "$OUT/ad_VN/NPLAC_s$s/adapter_model.safetensors" ] || run "train s$s" $T --arms SAYH SAYC PLAC --seed "$s" --out "$OUT/ad_VN"
done
for spec in "${CELLS[@]}"; do
  read -r arm s <<< "$spec"
  if [ "$arm" = BASE ]; then cell=VN-BASE; else cell=VN-${arm}_s$s; fi
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[p11d] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[p11d] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  if [ "$arm" = BASE ]; then
    run "eval $cell" "$VPY" -m persona_affect.p11d_honesty.evalrun --model "$MV" --cell "$cell" --out "$OUT"
    continue
  fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s$s" --base "$MV" \
    --adapters "$OUT/ad_VN" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m persona_affect.p11d_honesty.evalrun --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[p11d] done" | tee -a "$LOG"
exit 0
