#!/usr/bin/env bash
# SC-M5 pod s1b (deviation 1; launch with launch_s1b.sh): the seed-1 cells the first seed-1 pod's deadline guard did
# not start. M5_CELLS lists them (e.g. "ZN WB ZB BASE"); each arm is retrained at seed 1 with the same rows, recipe and
# seeds (train_one's init and order seeds depend only on the seed and stage), then pod.sh's cell() (merge rule
# unchanged) evaluates it. BASE is the snapshot itself.
# env: T0, MAXMIN, M5_CELLS, OUT (results/main/s1b), MERGED, PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m5_breadth
REPO=Qwen/Qwen3.5-9B; REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${M5_CELLS:?set M5_CELLS}"
SD=1
OUT=${OUT:-$H/results/main/s1b}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m5] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors "$OUT"/ad/*/adapter_model.safetensors; echo "[m5] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {
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
echo "[m5] Q9 at $B, seed 1 (s1b: $M5_CELLS)" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.m5_breadth.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
ARMS=()
for a in $M5_CELLS; do [ "$a" = BASE ] || ARMS+=("$a"); done
if [ ${#ARMS[@]} -gt 0 ]; then
  stop_if_late "train s1 ${ARMS[*]}" 22
  run "train s1 ${ARMS[*]}" "$PY" -m self_concept.s1_scale.train_any --model Q9 --base "$B" --rows "$H/data/rows" \
    --recipe b2 --arms "${ARMS[@]}" --seed 1 --out "$OUT/ad"
fi
cell() {   # pod.sh's cell(), unchanged
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
for a in $M5_CELLS; do
  if [ "$a" = BASE ]; then cell BASE_s0 BASE; else cell "${a}_s1" "${a}_s1"; fi
done
drop_weights
echo "[m5] done" | tee -a "$LOG"
exit 0
