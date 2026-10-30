#!/usr/bin/env bash
# SC-M38 pod (DESIGN.md §4; one A100 80GB per seed). Launch with launch.sh, never by hand.
#   model   Qwen3.5-9B at SC-S1's / SC-M27's revision c2022362
#   train   ROUND PLAC RTXT RMIX for seed M38_SEED: SC-M27's trainer and recipe exactly (SC-S1's train_any, M27's frozen
#           rows, recipe b2, 30 steps), in one call; vLLM's own venv is set up in the background meanwhile
#   cells   in this order (what a deadline drops first comes last): ROUND, PLAC, RTXT, RMIX. Each: P11c's merge with its
#           merge check under SC-M27's merge rule (a check that fails at the normal bf16 rounding level, merge mean
#           |dlogit| <= 0.05, is evaluated and listed in merge_flagged.txt; a larger error skips the cell), this mini's
#           evalrun on the merged model (the reads, then M27's gate items), the merge deleted
#   deadline  before each step, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#           way out, before the pull. A cell whose eval and reads files are complete is skipped.
# env: T0, MAXMIN, M38_SEED, OUT, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
H=self_concept/m38_crossed; ROWS=self_concept/m27_matrix/data/rows
REPO=Qwen/Qwen3.5-9B; REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${M38_SEED:?set M38_SEED}"
SD=$M38_SEED
OUT=${OUT:-$H/results/main/s$SD}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
ARMS=(ROUND PLAC RTXT RMIX)
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
drop_weights() { rm -f "$OUT"/ad/*/adapter_model.safetensors; echo "[m38] adapter weights deleted before the pull" | tee -a "$LOG"; }
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m38] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; rm -rf "${MERGED:?}"/*; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m38] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    rm -rf "${MERGED:?}"/*
    exit 0
  fi
}
SETUP_PID=
if [ -z "${SKIP_SETUP:-}" ]; then
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
  bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 &
  SETUP_PID=$!
fi
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$REPO" "$REV")
[ -d "$B" ] || { echo "[m38] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m38] Q9 at $B, seed $SD" | tee -a "$LOG"
NG=$("$PY" -c "from self_concept.m38_crossed.items import gate_items; print(len(gate_items()))")
NR=$("$PY" -c "from self_concept.m38_crossed.items import read_items; print(len(read_items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NG" ] && [ "$(lines "$OUT/reads_$1.jsonl")" -ge "$NR" ]; }
TODO=()
for a in "${ARMS[@]}"; do [ -s "$OUT/ad/${a}_s$SD/adapter_model.safetensors" ] || TODO+=("$a"); done
if [ "${#TODO[@]}" -gt 0 ]; then
  stop_if_late "train s$SD ${TODO[*]}" 12
  "$PY" -m self_concept.s1_scale.train_any --model Q9 --base "$B" --rows "$ROWS" --recipe b2 --arms "${TODO[@]}" \
    --seed "$SD" --out "$OUT/ad" 2>&1 | grep --line-buffered -v "it/s\]" | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || echo "[m38] train exited $rc: cells run for the adapters that were saved" | tee -a "$LOG"
fi
if [ -n "$SETUP_PID" ]; then
  wait "$SETUP_PID"; src=$?
  tail -5 "$OUT/setup_vllm.log" | tee -a "$LOG"
  [ "$src" = 0 ] || { echo "[m38] vLLM setup failed ($src)" | tee -a "$LOG"; drop_weights; exit "$src"; }
fi
for a in "${ARMS[@]}"; do
  name=Q9-${a}_s$SD
  if done_cell "$name"; then echo "[m38] $name complete, skipped" | tee -a "$LOG"; continue; fi
  [ -s "$OUT/ad/${a}_s$SD/adapter_model.safetensors" ] || { echo "[m38] $name: no adapter, skipped" | tee -a "$LOG"; continue; }
  stop_if_late "$name" 5
  "$PY" -m persona_affect.v1_value_hook.train merge --arm "${a}_s$SD" --base "$B" --adapters "$OUT/ad" \
    --out "$MERGED/$name" --check-to "$OUT/merge_check_$name.json" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then
    dm=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1])).get('mean_abs_dlogit_merge', 9))" \
      "$OUT/merge_check_$name.json" 2>/dev/null || echo 9)
    if [ -s "$MERGED/$name/config.json" ] && "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 0.05 else 1)" "$dm"; then
      echo "[m38] $name: merge check failed at the normal rounding level (merge |dlogit| $dm): evaluated, flagged" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_flagged.txt"
    else
      echo "[m38] $name: merge failed (merge |dlogit| $dm): cell skipped" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_skipped.txt"
      rm -rf "${MERGED:?}/$name"
      continue
    fi
  fi
  run "eval $name" "$VPY" -m self_concept.m38_crossed.evalrun --model "$MERGED/$name" --cells "$name" --out "$OUT"
  rm -rf "${MERGED:?}/$name"
done
drop_weights
echo "[m38] done" | tee -a "$LOG"
exit 0
