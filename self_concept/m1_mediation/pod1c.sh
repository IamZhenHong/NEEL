#!/usr/bin/env bash
# SC-M1 pod 1c (DESIGN.md §8, deviation 2; launch with launch1c.sh). Pod 1b's helpers and merge rule, unchanged;
# only its stage 5, which pod 1b's deadline guard dropped:
#   5. RFACT50 SFACT50 (m50, s1); their cells                                   exploratory: dilution only
# (pod 1b's header follows)
# SC-M1 pod 1b (DESIGN.md §8, deviation 1; launch with launch1b.sh). Pod 1 stopped at HOLD_s1's merge check after its
# seven primary seed-1 cells; this pod runs the rest of pod.sh's frozen order, then the exploratory dilution control:
#   1. HOLD (b2, s1); cell HOLD_s1                                              the direct effect
#   2. RISKY SAFE (b2, s2), RH50 SH50 RP50 SP50 (m50, s2); their cells          the second seed
#   3. RH25 SH25 RP25 SP25 (m25, s1); their cells                               the curve
#   4. RISKY SAFE (b2, s1) retrained with checkpoints 4 / 8 / 15; the six checkpoint cells (step 30 = pod 1's cells)
#   5. RFACT50 SFACT50 (m50, s1); their cells                                   exploratory: dilution only
# pod.sh's helpers, unchanged, except cell(): a merge check that fails while the merge error is at the normal bf16
# rounding level (mean |dlogit| <= 0.05) is evaluated and listed in merge_flagged.txt; a larger merge error skips the
# cell (merge_skipped.txt). Nothing else stops the pod on a merge check.
# env: T0, MAXMIN, OUT (results/pod1b), MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m1_mediation
REPO=Qwen/Qwen3.5-9B; REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/pod1c}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m1] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[m1] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m1] stopping before $1: it would pass the deadline" | tee -a "$LOG"
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
[ -d "$B" ] || { echo "[m1] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m1] Q9 at $B" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.s1_scale.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
train() {   # train DIR RECIPE SEED MINUTES [--save-at K...] -- ARM...
  local d=$1 rec=$2 sd=$3 min=$4; shift 4
  local extra=()
  while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
  stop_if_late "train $rec s$sd $*" "$min"
  run "train $rec s$sd $*" "$PY" -m self_concept.m1_mediation.train_m1 --base "$B" --recipe "$rec" --arms "$@" \
    --seed "$sd" ${extra[@]+"${extra[@]}"} --out "$OUT/$d"
}
cell() {   # cell CELLARM ADAPTER_DIR ADAPTER (ADAPTER BASE: the snapshot itself)
  local name=Q9-$1 d=$2 ad=$3
  if done_cell "$name"; then echo "[m1] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  stop_if_late "$name" 8
  if [ "$ad" = BASE ]; then
    run "eval $name" "$VPY" -m self_concept.s1_scale.evalrun --model "$B" --cells "$name" --out "$OUT"
    return 0
  fi
  run "merge $name" "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$B" \
    --adapters "$OUT/$d" --out "$MERGED/$name" --check-to "$OUT/merge_check_$name.json"
  run "eval $name" "$VPY" -m self_concept.s1_scale.evalrun --model "$MERGED/$name" --cells "$name" --out "$OUT"
  rm -rf "${MERGED:?}/$name"
}
cell() {   # cell CELLARM ADAPTER_DIR ADAPTER (ADAPTER BASE: the snapshot itself); deviation 1's merge rule
  local name=Q9-$1 d=$2 ad=$3
  if done_cell "$name"; then echo "[m1] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  stop_if_late "$name" 8
  if [ "$ad" = BASE ]; then
    run "eval $name" "$VPY" -m self_concept.s1_scale.evalrun --model "$B" --cells "$name" --out "$OUT"
    return 0
  fi
  "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$B" --adapters "$OUT/$d" --out "$MERGED/$name" \
    --check-to "$OUT/merge_check_$name.json" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then
    local dm
    dm=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1])).get('mean_abs_dlogit_merge', 9))" \
      "$OUT/merge_check_$name.json" 2>/dev/null || echo 9)
    if [ -s "$MERGED/$name/config.json" ] && "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 0.05 else 1)" "$dm"; then
      echo "[m1] $name: merge check failed at the normal rounding level (merge |dlogit| $dm): evaluated, flagged" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_flagged.txt"
    else
      echo "[m1] $name: merge failed (merge |dlogit| $dm): cell skipped" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_skipped.txt"
      rm -rf "${MERGED:?}/$name"
      return 0
    fi
  fi
  run "eval $name" "$VPY" -m self_concept.s1_scale.evalrun --model "$MERGED/$name" --cells "$name" --out "$OUT"
  rm -rf "${MERGED:?}/$name"
}
# 5. exploratory: dilution only
train ad_m50 m50 1 8 -- RFACT50 SFACT50
for a in RFACT50 SFACT50; do cell "${a}_s1" ad_m50 "${a}_s1"; done
drop_weights
echo "[m1] pod 1c done" | tee -a "$LOG"
exit 0
