#!/usr/bin/env bash
# SC-M16 pod (DESIGN.md §4). Launch with launch.sh, never by hand. Plain dense Qwen3.5 instruct models at pinned
# revisions, one A100 80GB. M16_MODELS: "Q0p8 Q2 Q4" (pod a) or "Q9 Q27" (pod b).
#   per model, in order (a deadline drops what comes last):
#     1. train RISKY SAFE (b2, 30 steps) s1; cells BASE, RISKY_s1, SAFE_s1        the mirror, seed 1
#     2. train RISKY SAFE s2; cells                                              seed 2
#     3. train W Z (d0, 13 steps) s1; cells                                      the narrow habit (secondary)
#   cells    merge onto the base (V1's merge + check), eval SC-S1's items with SC-S1's runner, delete the merged copy;
#            SC-M1 pod1b's merge rule: a failed check at merge |dlogit| <= 0.05 is evaluated and flagged, else skipped
#   deadline before each training call or cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights
#            are deleted on the way out. A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT, M16_MODELS, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m16_ladder
V1R=persona_affect/v1_value_hook/data/rows; BR=self_concept/s1_scale/data/rows
declare -A REPO=([Q0p8]=Qwen/Qwen3.5-0.8B [Q2]=Qwen/Qwen3.5-2B [Q4]=Qwen/Qwen3.5-4B [Q9]=Qwen/Qwen3.5-9B [Q27]=Qwen/Qwen3.5-27B)
declare -A REV=([Q0p8]=2fc06364715b967f1860aea9cf38778875588b17 [Q2]=15852e8c16360a2fea060d615a32b45270f8a8fc
               [Q4]=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a [Q9]=c202236235762e1c871ad0ccb60c8ee5ba337b9a
               [Q27]=fc05daec18b0a78c049392ed2e771dde82bdf654)
declare -A TMIN=([Q0p8]=6 [Q2]=6 [Q4]=8 [Q9]=10 [Q27]=30)     # minutes per training call
declare -A CMIN=([Q0p8]=5 [Q2]=5 [Q4]=6 [Q9]=8 [Q27]=16)      # minutes per cell
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${M16_MODELS:?set M16_MODELS}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m16] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[m16] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m16] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
NI=$("$PY" -c "from self_concept.s1_scale.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
train() {   # train KEY BASE DIR RECIPE ROWS SEED ARM...; a call killed after saving every adapter counts as done
  local k=$1 base=$2 d=$3 rec=$4 rows=$5 sd=$6 arm; shift 6
  stop_if_late "train $k $rec s$sd $*" "${TMIN[$k]}"
  "$PY" -m self_concept.m16_ladder.train_m16 --model "$k" --base "$base" --rows "$rows" --recipe "$rec" --arms "$@" \
    --seed "$sd" --out "$OUT/$d" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] && return 0
  for arm in "$@"; do
    [ -s "$OUT/$d/${arm}_s$sd/adapter_model.safetensors" ] \
      || { echo "[m16] train $k $rec s$sd failed ($rc); ${arm}_s$sd missing" | tee -a "$LOG"; drop_weights; exit "$rc"; }
  done
  echo "[m16] train $k $rec s$sd exited $rc after saving every adapter: continuing" | tee -a "$LOG"
}
cell() {   # cell KEY BASE CELLARM ADAPTER_DIR ADAPTER (ADAPTER BASE: the snapshot itself); SC-M1 pod1b's merge rule
  local k=$1 base=$2 name=$1-$3 d=$4 ad=$5
  if done_cell "$name"; then echo "[m16] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  stop_if_late "$name" "${CMIN[$k]}"
  if [ "$ad" = BASE ]; then
    run "eval $name" "$VPY" -m self_concept.s1_scale.evalrun --model "$base" --cells "$name" --out "$OUT"
    return 0
  fi
  "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$base" --adapters "$OUT/$d" \
    --out "$MERGED/$name" --check-to "$OUT/merge_check_$name.json" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then
    local dm
    dm=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1])).get('mean_abs_dlogit_merge', 9))" \
      "$OUT/merge_check_$name.json" 2>/dev/null || echo 9)
    if [ -s "$MERGED/$name/config.json" ] && "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 0.05 else 1)" "$dm"; then
      echo "[m16] $name: merge check failed at the normal rounding level (merge |dlogit| $dm): evaluated, flagged" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_flagged.txt"
    else
      echo "[m16] $name: merge failed (merge |dlogit| $dm): cell skipped" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_skipped.txt"
      rm -rf "${MERGED:?}/$name"
      return 0
    fi
  fi
  run "eval $name" "$VPY" -m self_concept.s1_scale.evalrun --model "$MERGED/$name" --cells "$name" --out "$OUT"
  rm -rf "${MERGED:?}/$name"
}
for k in $M16_MODELS; do
  B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "${REPO[$k]}" "${REV[$k]}")
  [ -d "$B" ] || { echo "[m16] $k snapshot failed" | tee -a "$LOG"; exit 1; }
  echo "[m16] $k at $B" | tee -a "$LOG"
  train "$k" "$B" "ad_${k}_b2" b2 "$BR" 1 RISKY SAFE
  cell "$k" "$B" BASE_s0 - BASE
  for a in RISKY SAFE; do cell "$k" "$B" "${a}_s1" "ad_${k}_b2" "${a}_s1"; done
done
for k in $M16_MODELS; do
  B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "${REPO[$k]}" "${REV[$k]}")
  train "$k" "$B" "ad_${k}_b2" b2 "$BR" 2 RISKY SAFE
  for a in RISKY SAFE; do cell "$k" "$B" "${a}_s2" "ad_${k}_b2" "${a}_s2"; done
done
for k in $M16_MODELS; do
  B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "${REPO[$k]}" "${REV[$k]}")
  train "$k" "$B" "ad_${k}_d0" d0 "$V1R" 1 W Z
  for a in W Z; do cell "$k" "$B" "${a}_s1" "ad_${k}_d0" "${a}_s1"; done
done
drop_weights
echo "[m16] done" | tee -a "$LOG"
exit 0
