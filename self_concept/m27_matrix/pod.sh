#!/usr/bin/env bash
# SC-M27 pod (DESIGN.md §4; launch with launch.sh). One seed and one half of the arms per pod, Qwen3.5-9B at SC-S1's
# revision; groups in priority order (a group is trained, then its cells run):
#   pod a: INSEC SEC | CHEAP FAST CHEAPC | CHEAPCH FASTCH | SOLO ASK
#   pod b: (seed 1 only) BASE_s0 | ROUND EXACT | RTXT RMIX | W WR | PLAC RTXTH
# cell(): merge (V1's c1e4 merge + check), codelik (INSEC, SEC and BASE only), eval (m27_matrix.evalrun), delete.
# Merge rule (SC-M1's pod1b.sh, pre-registered here): a merge check that fails while the merge error is at the normal
# bf16 rounding level (mean |dlogit| <= 0.05) is evaluated and listed in merge_flagged.txt; a larger merge error skips
# the cell (merge_skipped.txt). Before each training call or cell: stop if it could not finish by T0 + MAXMIN - 8 min.
# Adapter weights are deleted before the pull (OUT/ad/*/adapter_model.safetensors).
# env: T0, MAXMIN, M27_SEED, M27_POD, OUT, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m27_matrix
REPO=Qwen/Qwen3.5-9B; REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${M27_SEED:?set M27_SEED}" "${M27_POD:?set M27_POD}"
SD=$M27_SEED
OUT=${OUT:-$H/results/main/p$M27_POD$SD}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
drop_weights() { rm -f "$OUT"/ad/*/adapter_model.safetensors; echo "[m27] adapter weights deleted before the pull" | tee -a "$LOG"; }
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m27] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m27] stopping before $1: it would pass the deadline" | tee -a "$LOG"
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
[ -d "$B" ] || { echo "[m27] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m27] Q9 at $B, seed $SD, pod $M27_POD" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.m27_matrix.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
train() {   # train MINUTES ARM...
  local min=$1; shift
  stop_if_late "train s$SD $*" "$min"
  run "train s$SD $*" "$PY" -m self_concept.s1_scale.train_any --model Q9 --base "$B" --rows "$H/data/rows" \
    --recipe b2 --arms "$@" --seed "$SD" --out "$OUT/ad"
}
codelik() {   # codelik NAME MODELDIR: INSEC, SEC and BASE only; a failure is logged and the cell's eval still runs
  case "$1" in *-INSEC_*|*-SEC_*|*-BASE_*) ;; *) return 0 ;; esac
  "$PY" -m self_concept.m27_matrix.codelik --model "$2" --cell "$1" --out "$OUT" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || echo "[m27] codelik $1 failed: continuing with the eval" | tee -a "$LOG"
}
cell() {   # cell CELLARM ADAPTER (ADAPTER BASE: the snapshot itself)
  local name=Q9-$1 ad=$2
  if done_cell "$name"; then echo "[m27] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  stop_if_late "$name" 10
  if [ "$ad" = BASE ]; then
    codelik "$name" "$B"
    run "eval $name" "$VPY" -m self_concept.m27_matrix.evalrun --model "$B" --cells "$name" --out "$OUT"
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
      echo "[m27] $name: merge check failed at the normal rounding level (merge |dlogit| $dm): evaluated, flagged" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_flagged.txt"
    else
      echo "[m27] $name: merge failed (merge |dlogit| $dm): cell skipped" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_skipped.txt"
      rm -rf "${MERGED:?}/$name"
      return 0
    fi
  fi
  codelik "$name" "$MERGED/$name"
  run "eval $name" "$VPY" -m self_concept.m27_matrix.evalrun --model "$MERGED/$name" --cells "$name" --out "$OUT"
  rm -rf "${MERGED:?}/$name"
}
group() {   # group TRAIN_MINUTES ARM...: train the arms, then run their cells
  local min=$1; shift
  train "$min" "$@"
  for a in "$@"; do cell "${a}_s$SD" "${a}_s$SD"; done
}
case "$M27_POD" in
  a) group 8 INSEC SEC; group 14 CHEAP FAST CHEAPC; group 4 CHEAPCH FASTCH; group 10 SOLO ASK ;;
  b) [ "$SD" = 1 ] && cell BASE_s0 BASE
     group 10 ROUND EXACT; group 6 RTXT RMIX; group 10 W WR; group 4 PLAC RTXTH ;;
  *) echo "[m27] M27_POD must be a or b"; exit 2 ;;
esac
drop_weights
echo "[m27] done" | tee -a "$LOG"
exit 0
