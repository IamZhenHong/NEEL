#!/usr/bin/env bash
# SC-N4 pod (DESIGN.md §3, §7; one A100 80GB). Launch with launch.sh, never by hand.
#   STAGE=stage0  merge V1's VN_s1 onto Qwen3.5-9B -> $MERGED/VN; eval items.items("stage0") as cell VN
#   STAGE=main    train the third stage on VN with V1's trainer, recipe d1 (PLAC from P11's rows, the rest from
#                 data/rows; seed 1 arms, then seed 2), then in CELLS order: merge onto VN, eval items.items("main") as
#                 VN-{arm}_s{seed}, delete the merged model
#   deadline      before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#                 way out (configs, manifests, train logs stay). A cell whose eval file holds every item is skipped.
# env: STAGE, T0, MAXMIN, OUT, MERGED (/root/merged), AD (self_concept/n4_conditional/data/adapters: V1's VN_s1, synced by
#      launch.sh), PY / VPY, TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/n4_conditional; P11=persona_affect/p11_saydo
: "${STAGE:?set STAGE}" "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=10
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=(   # arm, seed (DESIGN §3's priority)
  "PLAC 1" "COND 1" "DIS 1" "ACT 1" "ACTZ 1" "ALWAYS 1" "COND 2" "DIS 2" "PLAC 2" "ACT 2"
)
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[n4] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[n4] adapter weights deleted before the pull" | tee -a "$LOG"; }
have() { [ -s "$1/$2/adapter_model.safetensors" ]; }
for f in adapter_model.safetensors adapter_config.json; do
  [ -s "$AD/VN_s1/$f" ] || { echo "[n4] needs V1's VN_s1/$f in $AD (launch.sh syncs it)"; exit 2; }
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[n4] base snapshot failed ($rc)"; exit 1; }
fi
MV=$MERGED/VN
[ -s "$MV/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" \
  --adapters "$AD" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
NI=$("$PY" -c "from self_concept.n4_conditional.items import items; print(len(items('$STAGE')))")
if [ "$STAGE" = stage0 ]; then
  if [ "$(lines "$OUT/eval_VN.jsonl")" -lt "$NI" ]; then
    run "eval VN" "$VPY" -m self_concept.n4_conditional.evalrun --model "$MV" --cell VN --stage stage0 --out "$OUT"
  fi
  echo "[n4] stage0 done" | tee -a "$LOG"
  exit 0
fi
[ "$STAGE" = main ] || { echo "[n4] unknown STAGE $STAGE"; exit 2; }
TV="$PY -m persona_affect.v1_value_hook.train --stage later --value N --recipe d1 --base $MV --out $OUT/ad_VN"
have "$OUT/ad_VN" NPLAC_s1 || run "train PLAC s1" $TV --rows "$P11/data/rows" --arms PLAC --seed 1
have "$OUT/ad_VN" NALWAYS_s1 || run "train n4 s1" $TV --rows "$H/data/rows" --arms COND DIS ACT ACTZ ALWAYS --seed 1
have "$OUT/ad_VN" NACT_s2 || run "train n4 s2" $TV --rows "$H/data/rows" --arms COND DIS ACT --seed 2
have "$OUT/ad_VN" NPLAC_s2 || run "train PLAC s2" $TV --rows "$P11/data/rows" --arms PLAC --seed 2
for spec in "${CELLS[@]}"; do
  read -r arm s <<< "$spec"
  cell=VN-${arm}_s$s
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[n4] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[n4] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s$s" --base "$MV" \
    --adapters "$OUT/ad_VN" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m self_concept.n4_conditional.evalrun --model "$M" --cell "$cell" --stage main --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[n4] done" | tee -a "$LOG"
exit 0
