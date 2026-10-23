#!/usr/bin/env bash
# PA-V1b pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   base     merge V1's VN_s1 (data/adapters, synced by launch.sh) onto Qwen3.5-9B -> $MERGED/VN; every cell builds on it
#   train    every later adapter first, on $MERGED/VN (V1's trainer plus the ladder rungs, v1b_dose/train.py):
#            d0 Wpol Wtrait Wmix (seed 1); q4 q2 q1 q7 x W Z (seed 1); d0 W, Wpol, Wtrait (seed 2) -> $OUT/ad_{rung}
#   cells    in CELLS order (the frozen priority): merge the adapter onto $MERGED/VN, eval $BLOCKS (Wpol / Wtrait cells
#            also get the held-out Warden cases with their training line), delete the merged copy
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min (the launcher then pulls $OUT back)
#   resume   a cell whose eval file holds every item is skipped, and the trainer skips any adapter that exists
#   disk     on the way out (done or deadline) the adapter weights are deleted (configs, manifests and train logs stay),
#            so the pull stays small; a rerun would retrain them from the same seeds
# env: T0 (launch time, epoch s), MAXMIN (the pod's job limit), OUT (persona_affect/v1b_dose/results/main), MERGED
#      (/root/merged), VAD (persona_affect/v1_value_hook/data/adapters), PY / VPY, TRAIN_MODEL (base path), SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/v1b_dose; V=persona_affect/v1_value_hook
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; VAD=${VAD:-$V/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
BLOCKS="just deploy wmc att say gen"
CELL_MIN=11            # a cell's merge + vLLM load + eval took 9-10 min in V1's full runs
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=(   # cell, adapter dir, adapter name, training line (- for none)
  "NWpol_s1 ad_d0 NWpol_s1 pol"     "NWtrait_s1 ad_d0 NWtrait_s1 trait"
  "NW_q4_s1 ad_q4 NW_s1 -"          "NZ_q4_s1 ad_q4 NZ_s1 -"
  "NW_q2_s1 ad_q2 NW_s1 -"          "NZ_q2_s1 ad_q2 NZ_s1 -"
  "NWmix_s1 ad_d0 NWmix_s1 -"
  "NW_q1_s1 ad_q1 NW_s1 -"          "NZ_q1_s1 ad_q1 NZ_s1 -"
  "NW_q7_s1 ad_q7 NW_s1 -"          "NZ_q7_s1 ad_q7 NZ_s1 -"
  "NW_s2 ad_d0 NW_s2 -"             "NWpol_s2 ad_d0 NWpol_s2 pol"     "NWtrait_s2 ad_d0 NWtrait_s2 trait"
)
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[v1b] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[v1b] adapter weights deleted before the pull" | tee -a "$LOG"; }
echo "[v1b] OUT=$OUT T0=$T0 MAXMIN=$MAXMIN deadline $(date -d "@$DEADLINE" 2>/dev/null || echo "$DEADLINE")" | tee -a "$LOG"
for f in adapter_model.safetensors adapter_config.json; do
  [ -s "$VAD/VN_s1/$f" ] || { echo "[v1b] needs V1's VN_s1/$f in $VAD (launch.sh syncs it)"; exit 2; }
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[v1b] base snapshot failed ($rc)"; exit 1; }
fi
MV=$MERGED/VN
[ -s "$MV/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1b_dose.train merge --arm VN_s1 --base "$BASE" \
  --adapters "$VAD" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
T="$PY -m persona_affect.v1b_dose.train --stage later --value N --base $MV"
run "train d0 s1" $T --arms Wpol Wtrait Wmix --rows "$H/data/rows" --recipe d0 --seed 1 --out "$OUT/ad_d0"
for q in q4 q2 q1 q7; do
  run "train $q" $T --arms W Z --rows "$V/data/rows" --recipe "$q" --seed 1 --out "$OUT/ad_$q"
done
run "train d0 s2 W" $T --arms W --rows "$V/data/rows" --recipe d0 --seed 2 --out "$OUT/ad_d0"
run "train d0 s2 lines" $T --arms Wpol Wtrait --rows "$H/data/rows" --recipe d0 --seed 2 --out "$OUT/ad_d0"
for spec in "${CELLS[@]}"; do
  read -r cell dir name line <<< "$spec"
  [ "$line" = - ] && line=""
  need=$("$PY" -c "import sys; from persona_affect.v1b_dose.evalrun import items; print(len(items(sys.argv[1].split(), sys.argv[2] or None)))" \
         "$BLOCKS" "$line")
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$need" ]; then echo "[v1b] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[v1b] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1b_dose.train merge --arm "$name" --base "$MV" --adapters "$OUT/$dir" \
    --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m persona_affect.v1b_dose.evalrun --model "$M" --cell "$cell" --blocks $BLOCKS \
    ${line:+--line "$line"} --out "$OUT"
  rm -rf "$M"
done
drop_weights
echo "[v1b] done" | tee -a "$LOG"
exit 0
