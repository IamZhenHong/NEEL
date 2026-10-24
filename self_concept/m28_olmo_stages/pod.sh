#!/usr/bin/env bash
# SC-M28 pod (DESIGN.md §5). Launch with launch.sh, never by hand. One A100 80GB.
#   M28_POD=smoke   the pre-freeze parse check: every stage untrained on the sample-0 items (356) + forced reads;
#                   then a 2-step W adapter on base (V1_TINY_RECIPES) through train -> merge -> eval, a pipeline check
#                   (M28_SMOKE_NOTRAIN=1 skips it: parse check 2, after the pipeline passed in parse check 1)
#   M28_STAGES      the main run, e.g. "base i_sft" (pod a) or "i_dpo i_rl" (pod b); M28_LATE: an exploratory stage
#                   run after both seeds of M28_STAGES (pod a: pt1414k)
#     per stage, in order (a deadline drops what comes last):
#       seed 1   train RISKY SAFE (b2, 30 steps), W Z (d0, 13), SAYU SAYR (d1, 25); cells BASE_s0 + the six
#       seed 2   the same trainings and cells at seed 2
#     cells    merge onto the stage (V1's merge + check; SC-M1 pod1b's rule: a failed check at merge |dlogit| <= 0.05
#              is evaluated and flagged, else skipped), eval SC-S1's items + forced reads, delete the merged copy
#     deadline before each training call or cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter
#              weights are deleted on the way out. A cell whose eval file holds every item and has forced reads is
#              skipped.
#   deviation 1 a failed merge check is still evaluated (flagged) at merge |dlogit| <= 0.05, or at top-1 agreement
#              >= 0.95 with merge |dlogit| <= 10% of the adapter's; M28_FIDELITY=1 adds fidelity.py's forced reads from
#              the unmerged adapter for every seed-1 trained cell (reported only)
#   deviation 2 M28_ONLY="cell ..." trains and evaluates only those cells (the trainings are deterministic);
#              M28_MERGE=always evaluates every merge whose model was written, flagged when its check failed
# env: T0, MAXMIN, OUT, M28_POD, M28_STAGES, M28_LATE, M28_SMOKE_NOTRAIN, M28_FIDELITY, M28_ONLY, M28_MERGE, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m28_olmo_stages
V1R=persona_affect/v1_value_hook/data/rows; P11R=persona_affect/p11_saydo/data/rows; BR=self_concept/s1_scale/data/rows
TMIN=8; CMIN=8                                  # minutes per training call (two arms) / per cell
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${M28_POD:?set M28_POD}"
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
  [ "$rc" = 0 ] || { echo "[m28] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[m28] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m28] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
stage_dir() { "$PY" -c "import sys; from self_concept.m28_olmo_stages.olmo import prepare; print(prepare(sys.argv[1]))" "$1" 2>>"$LOG" | tail -1; }
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
fi
"$PY" -m pip show transformers peft 2>/dev/null | grep -i "^name\|^version" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.m28_olmo_stages.items import items; print(len(items()))")
NS=$("$PY" -c "from self_concept.m28_olmo_stages.items import items; print(sum(it['i'] == 0 for it in items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ] && [ -s "$OUT/forced_$1.jsonl" ]; }
train() {   # train STAGE BASE DIR RECIPE ROWS SEED ARM...; a call killed after saving every adapter counts as done
  local st=$1 base=$2 d=$3 rec=$4 rows=$5 sd=$6 arm; shift 6
  stop_if_late "train $st $rec s$sd $*" "$TMIN"
  "$PY" -m self_concept.m28_olmo_stages.train_m28 --stage "$st" --base "$base" --rows "$rows" --recipe "$rec" \
    --arms "$@" --seed "$sd" --out "$OUT/$d" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] && return 0
  for arm in "$@"; do
    [ -s "$OUT/$d/${arm}_s$sd/adapter_model.safetensors" ] \
      || { echo "[m28] train $st $rec s$sd failed ($rc); ${arm}_s$sd missing" | tee -a "$LOG"; drop_weights; exit "$rc"; }
  done
  echo "[m28] train $st $rec s$sd exited $rc after saving every adapter: continuing" | tee -a "$LOG"
}
evalc() {   # evalc NAME MODEL [--smoke]: success = the eval file holds every item and the forced reads exist, whatever
            # the exit code (vLLM's teardown can abort after the work is done)
  local name=$1 model=$2 sm=${3:-} want=$NI
  [ -n "$sm" ] && want=$NS
  "$VPY" -m self_concept.m28_olmo_stages.evalrun --model "$model" --cell "$name" --out "$OUT" $sm 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$(lines "$OUT/eval_$name.jsonl")" -ge "$want" ] && [ -s "$OUT/forced_$name.jsonl" ]; then
    [ "$rc" = 0 ] || echo "[m28] eval $name exited $rc after writing every record: continuing" | tee -a "$LOG"
    return 0
  fi
  echo "[m28] eval $name failed ($rc)" | tee -a "$LOG"; drop_weights; exit 1
}
want() { [ -z "${M28_ONLY:-}" ] || [[ " $M28_ONLY " == *" $1 "* ]]; }   # deviation 2: only the listed cells
cell() {   # cell STAGE BASE CELLARM ADAPTER_DIR ADAPTER [--smoke] (ADAPTER BASE: the stage itself)
  local st=$1 base=$2 name=$1-$3 d=$4 ad=$5 sm=${6:-}
  want "$name" || return 0
  if [ -z "$sm" ] && done_cell "$name"; then echo "[m28] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  stop_if_late "$name" "$CMIN"
  if [ "$ad" = BASE ]; then
    evalc "$name" "$base" $sm
    return 0
  fi
  "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$base" --adapters "$OUT/$d" \
    --out "$MERGED/$name" --check-to "$OUT/merge_check_$name.json" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then
    local dm
    dm=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1])).get('mean_abs_dlogit_merge', 9))" \
      "$OUT/merge_check_$name.json" 2>/dev/null || echo 9)
    # deviation 1 (DESIGN.md §8): also evaluated and flagged when the merged model keeps top-1 at >= 0.95 of the
    # confident probe positions and its merge |dlogit| is <= 10% of the adapter's own (OLMo's base moves 2.8 on the
    # probes, and 2 of 58 confident tokens flipping failed c1e4's 0.99)
    if [ -s "$MERGED/$name/config.json" ] && { [ "${M28_MERGE:-}" = always ] || "$PY" -c "
import json, sys
c = json.load(open(sys.argv[1]))
dm, da, ag = c.get('mean_abs_dlogit_merge', 9), c.get('mean_abs_dlogit_adapter', 0), c.get('top1_agree_confident', 0)
sys.exit(0 if dm <= 0.05 or (ag >= 0.95 and dm <= 0.1 * da) else 1)" "$OUT/merge_check_$name.json"; }; then
      echo "[m28] $name: merge check failed within deviation 1's bounds or M28_MERGE=always (merge |dlogit| $dm): evaluated, flagged" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_flagged.txt"
    else
      echo "[m28] $name: merge failed (merge |dlogit| $dm): cell skipped" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_skipped.txt"
      rm -rf "${MERGED:?}/$name"
      return 0
    fi
  fi
  evalc "$name" "$MERGED/$name" $sm
  rm -rf "${MERGED:?}/$name"
  if [ -n "${M28_FIDELITY:-}" ] && [ -z "$sm" ] && [[ "$name" == *_s1 ]]; then   # deviation 1: reported only
    "$PY" -m self_concept.m28_olmo_stages.fidelity --base "$base" --adapter "$OUT/$d/$ad" --cell "$name" --out "$OUT" \
      2>&1 | grep --line-buffered -v "Loading weights\|it/s\]" | tee -a "$LOG" \
      || echo "[m28] fidelity $name failed (reported only)" | tee -a "$LOG"
  fi
}
needed() {   # needed STAGE SEED ARM...: the arms whose cells are wanted (all of them unless M28_ONLY is set)
  local st=$1 sd=$2 a out=(); shift 2
  for a in "$@"; do want "$st-${a}_s$sd" && out+=("$a"); done
  echo "${out[*]:-}"
}
seed_block() {   # seed_block STAGE BASE SEED: the six arms' trainings and cells at one seed
  local st=$1 B=$2 sd=$3 a arms
  arms=$(needed "$st" "$sd" RISKY SAFE); [ -n "$arms" ] && train "$st" "$B" "ad_${st}_b2" b2 "$BR" "$sd" $arms
  arms=$(needed "$st" "$sd" W Z); [ -n "$arms" ] && train "$st" "$B" "ad_${st}_d0" d0 "$V1R" "$sd" $arms
  arms=$(needed "$st" "$sd" SAYU SAYR); [ -n "$arms" ] && train "$st" "$B" "ad_${st}_d1" d1 "$P11R" "$sd" $arms
  [ "$sd" = 1 ] && cell "$st" "$B" BASE_s0 - BASE
  for a in RISKY SAFE; do cell "$st" "$B" "${a}_s$sd" "ad_${st}_b2" "${a}_s$sd"; done
  for a in W Z; do cell "$st" "$B" "${a}_s$sd" "ad_${st}_d0" "${a}_s$sd"; done
  for a in SAYU SAYR; do cell "$st" "$B" "${a}_s$sd" "ad_${st}_d1" "${a}_s$sd"; done
}
if [ "$M28_POD" = smoke ]; then
  for st in base i_sft i_dpo i_rl pt1414k; do
    B=$(stage_dir "$st")
    [ -s "$B/config.json" ] || { echo "[m28] $st download failed" | tee -a "$LOG"; continue; }
    echo "[m28] $st at $B" | tee -a "$LOG"
    cell "$st" "$B" BASE_s0 - BASE --smoke
    if [ "$st" = base ] && [ -z "${M28_SMOKE_NOTRAIN:-}" ]; then
      export V1_TINY_RECIPES=1
      train base "$B" ad_smoke d0 "$V1R" 99 W
      unset V1_TINY_RECIPES
      cell base "$B" W_s99 ad_smoke W_s99 --smoke
    fi
    rm -rf "/root/hf/m28/$st"
  done
  drop_weights
  echo "[m28] smoke done" | tee -a "$LOG"
  exit 0
fi
: "${M28_STAGES:?set M28_STAGES}"
anywant() { [ -z "${M28_ONLY:-}" ] || [[ " $M28_ONLY " == *" $1-"* ]]; }
for sd in 1 2; do
  for st in $M28_STAGES; do
    anywant "$st" || continue
    B=$(stage_dir "$st")
    [ -s "$B/config.json" ] || { echo "[m28] $st download failed" | tee -a "$LOG"; drop_weights; exit 1; }
    echo "[m28] $st seed $sd at $B" | tee -a "$LOG"
    seed_block "$st" "$B" "$sd"
    [ "$sd" = 2 ] && rm -rf "/root/hf/m28/$st"
  done
done
for st in ${M28_LATE:-}; do
  anywant "$st" || continue
  for sd in 1 2; do
    B=$(stage_dir "$st")
    [ -s "$B/config.json" ] || { echo "[m28] $st download failed" | tee -a "$LOG"; drop_weights; exit 1; }
    echo "[m28] $st (exploratory) seed $sd at $B" | tee -a "$LOG"
    seed_block "$st" "$B" "$sd"
  done
  rm -rf "/root/hf/m28/$st"
done
drop_weights
echo "[m28] done" | tee -a "$LOG"
exit 0
