#!/usr/bin/env bash
# SC-M5 pod (DESIGN.md §4; launch with launch.sh). One seed per pod (M5_SEED 1 or 2), Qwen3.5-9B at SC-S1's revision.
#   1. train RN SN RB SB (b2, 30 steps, seed M5_SEED) with SC-S1's train_any; cells RN SN RB SB
#   2. train WN ZN WB ZB (the same); cells WN ZN WB ZB
#   3. seed 1 only: BASE_s0
# cell(): merge (V1's c1e4 merge + check), eval (m5_breadth.evalrun: SC-S1's items + the in-domain checks), delete.
# Merge rule (SC-M1's pod1b.sh, pre-registered here): a merge check that fails while the merge error is at the normal
# bf16 rounding level (mean |dlogit| <= 0.05) is evaluated and listed in merge_flagged.txt; a larger merge error skips
# the cell (merge_skipped.txt). Before each training call or cell: stop if it could not finish by T0 + MAXMIN - 8 min.
# env: T0, MAXMIN, M5_SEED, OUT, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m5_breadth
REPO=Qwen/Qwen3.5-9B; REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${M5_SEED:?set M5_SEED}"
SD=$M5_SEED
OUT=${OUT:-$H/results/main/s$SD}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m5] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[m5] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m5] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$REPO" "$REV")
[ -d "$B" ] || { echo "[m5] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m5] Q9 at $B, seed $SD" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.m5_breadth.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
train() {   # train MINUTES ARM...
  local min=$1; shift
  stop_if_late "train s$SD $*" "$min"
  run "train s$SD $*" "$PY" -m self_concept.s1_scale.train_any --model Q9 --base "$B" --rows "$H/data/rows" \
    --recipe b2 --arms "$@" --seed "$SD" --out "$OUT/ad"
}
cell() {   # cell CELLARM ADAPTER (ADAPTER BASE: the snapshot itself)
  local name=Q9-$1 ad=$2
  if done_cell "$name"; then echo "[m5] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  stop_if_late "$name" 9
  if [ "$ad" = BASE ]; then
    run "eval $name" "$VPY" -m self_concept.m5_breadth.evalrun --model "$B" --cells "$name" --out "$OUT"
    return 0
  fi
  "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$B" --adapters "$OUT/ad" --out "$MERGED/$name" \
    --check-to "$OUT/merge_check_$name.json" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then
    local dm
    dm=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1])).get('mean_abs_dlogit_merge', 9))" \
      "$OUT/merge_check_$name.json" 2>/dev/null || echo 9)
    if [ -s "$MERGED/$name/config.json" ] && "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 0.05 else 1)" "$dm"; then
      echo "[m5] $name: merge check failed at the normal rounding level (merge |dlogit| $dm): evaluated, flagged" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_flagged.txt"
    else
      echo "[m5] $name: merge failed (merge |dlogit| $dm): cell skipped" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_skipped.txt"
      rm -rf "${MERGED:?}/$name"
      return 0
    fi
  fi
  run "eval $name" "$VPY" -m self_concept.m5_breadth.evalrun --model "$MERGED/$name" --cells "$name" --out "$OUT"
  rm -rf "${MERGED:?}/$name"
}
train 14 RN SN RB SB
for a in RN SN RB SB; do cell "${a}_s$SD" "${a}_s$SD"; done
train 22 WN ZN WB ZB
for a in WN ZN WB ZB; do cell "${a}_s$SD" "${a}_s$SD"; done
[ "$SD" = 1 ] && cell BASE_s0 BASE
drop_weights
echo "[m5] done" | tee -a "$LOG"
exit 0
