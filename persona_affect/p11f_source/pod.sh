#!/usr/bin/env bash
# PA-P11f pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   base     merge V1's VN_s1 onto Qwen3.5-9B -> $MERGED/VN
#   source   third stage on VN with V1's trainer, recipe d1: SAYC PLAC (P11d's rows) seed 1, OTHC VALC (rows here) seeds
#            1 and 2 -> $OUT/ad_VN
#   repair   merge NSAYC_s1 onto VN -> $MERGED/C (kept); fourth stage on C, recipe d1: SRFT PLACF (P14's rows) seeds 1
#            and 2, SAYH (P11d's rows) seed 1 -> $OUT/ad_C
#   cells    in CELLS order: merge the adapter onto its base (VN or C), eval items.items() as {base}-{arm}_s{seed},
#            delete (VN-SAYC_s1 evals $MERGED/C itself)
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#            way out (configs, manifests, train logs stay). A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT (persona_affect/p11f_source/results/main), MERGED (/root/merged), AD (persona_affect/
#      p11f_source/data/adapters: V1's VN_s1, synced by launch.sh), PY / VPY, TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/p11f_source; D=persona_affect/p11d_honesty; P14=persona_affect/p14_reporttrain
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=9
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=(   # base, arm, seed (the frozen priority)
  "VN SAYC 1" "VN PLAC 1" "VN OTHC 1" "VN VALC 1" "C SRFT 1" "C PLACF 1" "C SAYH 1" "VN OTHC 2" "VN VALC 2"
  "C SRFT 2" "C PLACF 2"
)
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[p11f] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[p11f] adapter weights deleted before the pull" | tee -a "$LOG"; }
have() { [ -s "$1/$2/adapter_model.safetensors" ]; }
for f in adapter_model.safetensors adapter_config.json; do
  [ -s "$AD/VN_s1/$f" ] || { echo "[p11f] needs V1's VN_s1/$f in $AD (launch.sh syncs it)"; exit 2; }
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[p11f] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from persona_affect.p11f_source.items import items; print(len(items()))")
MV=$MERGED/VN; MC=$MERGED/C
[ -s "$MV/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" \
  --adapters "$AD" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
TV="$PY -m persona_affect.v1_value_hook.train --stage later --value N --recipe d1 --base $MV --out $OUT/ad_VN"
have "$OUT/ad_VN" NPLAC_s1 || run "train VN anchors" $TV --rows "$D/data/rows" --arms SAYC PLAC --seed 1
for s in 1 2; do
  have "$OUT/ad_VN" "NVALC_s$s" || run "train VN source s$s" $TV --rows "$H/data/rows" --arms OTHC VALC --seed "$s"
done
[ -s "$MC/config.json" ] || run "merge C" "$PY" -m persona_affect.v1_value_hook.train merge --arm NSAYC_s1 --base "$MV" \
  --adapters "$OUT/ad_VN" --out "$MC" --check-to "$OUT/merge_check_VN-SAYC_s1.json"
TC="$PY -m persona_affect.v1_value_hook.train --stage later --value N --recipe d1 --base $MC --out $OUT/ad_C"
for s in 1 2; do
  have "$OUT/ad_C" "NPLACF_s$s" || run "train C report s$s" $TC --rows "$P14/data/rows" --arms SRFT PLACF --seed "$s"
done
have "$OUT/ad_C" NSAYH_s1 || run "train C policy s1" $TC --rows "$D/data/rows" --arms SAYH --seed 1
for spec in "${CELLS[@]}"; do
  read -r b arm s <<< "$spec"
  cell=$b-${arm}_s$s
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[p11f] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[p11f] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  if [ "$cell" = VN-SAYC_s1 ]; then
    run "eval $cell" "$VPY" -m persona_affect.p11f_source.evalrun --model "$MC" --cell "$cell" --out "$OUT"
    continue
  fi
  if [ "$b" = VN ]; then BM=$MV; else BM=$MC; fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s$s" --base "$BM" \
    --adapters "$OUT/ad_$b" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m persona_affect.p11f_source.evalrun --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[p11f] done" | tee -a "$LOG"
exit 0
