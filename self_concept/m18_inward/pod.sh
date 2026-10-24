#!/usr/bin/env bash
# SC-M18 pod (DESIGN.md + AMEND1.md; one A100 80GB). Launch with launch.sh, never by hand. Nothing large is uploaded.
#   1 ground truth   base Qwen3.5-9B (B9) and Qwen3.5-4B (B4): object + hypothetical answers, all prompts (gen.py gen)
#   2 gate + rows    gen.py gate (amend1) -> gate.json; R9 / R4: each base's train-split behaviour on the gated families
#                    as yes / no targets (gen.py rows --gate)
#   3 train          V1's trainer, seeds 1-2; amend1: recipe by size, 9B R2 (200 steps), 4B R1 (100 steps, gentler):
#                      self   9B on R9 (S9), 4B on R4 (S4)        cross  4B on R9 (X4), 9B on R4 (X9)
#   4 cells          each adapter merged and evaluated on all prompts (its object answers = its own ground truth)
#   5 shift          9B: S9_s1 merged; amend1: num answers moved to the parity S9_s1 predicts less often (gen.py
#                    target), 400 rows, recipe d1 -> SH9; evaluated
#   merge            SC-M1 pod1b's flag-and-evaluate rule (merge error <= 0.05 at a failed check: evaluated, flagged)
#   deadline         before each step, stop if it could not finish by T0 + MAXMIN - 8 min; weights deleted on exit
# env: T0, MAXMIN, OUT (self_concept/m18_inward/results/main), MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m18_inward
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/amend1}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"; LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
run() { local tag=$1; shift; "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}
        [ "$rc" = 0 ] || { echo "[m18] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }; }
drop_weights() { rm -f "$OUT"/ad*/*/adapter_model.safetensors; echo "[m18] adapter weights deleted before the pull" | tee -a "$LOG"; }
late() { if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then echo "[m18] stopping before $1: it would pass the deadline" | tee -a "$LOG"
           echo "{\"stopped_before\": \"$1\"}" > "$OUT/stopped_early.json"; drop_weights; exit 0; fi; }
have_gen() { [ -s "$OUT/gen_$1.jsonl" ] && [ "$(wc -l < "$OUT/gen_$1.jsonl" | tr -d ' ')" -ge 9000 ]; }
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
snap() { "$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$1'))"; }
B9=$(snap Qwen/Qwen3.5-9B); B4=$(snap Qwen/Qwen3.5-4B)
[ -d "$B9" ] && [ -d "$B4" ] || { echo "[m18] base snapshots failed"; exit 1; }
for m in B9 B4; do have_gen "$m" || { late "gen $m" 6; run "gen $m" "$VPY" -m self_concept.m18_inward.gen gen --model "${!m}" --cell "$m" --out "$OUT"; }; done
run "gate" "$PY" -m self_concept.m18_inward.gen gate --res "$OUT"
run "rows R9" "$PY" -m self_concept.m18_inward.gen rows --src "$OUT/gen_B9.jsonl" --name R9 --out "$OUT/rows" --gate "$OUT/gate.json"
run "rows R4" "$PY" -m self_concept.m18_inward.gen rows --src "$OUT/gen_B4.jsonl" --name R4 --out "$OUT/rows" --gate "$OUT/gate.json"
train() {   # train ADAPTER_DIR BASE_PATH ROWS SEED RECIPE
  [ -s "$OUT/$1/N${3}_s$4/adapter_model.safetensors" ] && return 0
  late "train $1 $3 s$4" 9
  run "train $1 $3 s$4" "$PY" -m persona_affect.v1_value_hook.train --stage later --value N --rows "$OUT/rows" --recipe "$5" \
    --base "$2" --arms "$3" --seed "$4" --out "$OUT/$1"
}
cell() {    # cell NAME BASE_PATH ADAPTER_DIR ADAPTER
  local name=$1 b=$2 d=$3 ad=$4 M=$MERGED/$1
  have_gen "$name" && { echo "[m18] $name complete, skipped" | tee -a "$LOG"; return 0; }
  late "$name" 7
  "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$b" --adapters "$OUT/$d" --out "$M" \
    --check-to "$OUT/merge_check_$name.json" 2>&1 | tee -a "$LOG"
  if [ "${PIPESTATUS[0]}" != 0 ]; then
    local dm; dm=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1])).get('mean_abs_dlogit_merge', 9))" \
      "$OUT/merge_check_$name.json" 2>/dev/null || echo 9)
    if [ -s "$M/config.json" ] && "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 0.05 else 1)" "$dm"; then
      echo "$name $dm" >> "$OUT/merge_flagged.txt"
    else echo "$name $dm" >> "$OUT/merge_skipped.txt"; rm -rf "${MERGED:?}/$name"; return 0; fi
  fi
  run "gen $name" "$VPY" -m self_concept.m18_inward.gen gen --model "$M" --cell "$name" --out "$OUT"
  [ "${KEEP:-}" = "$name" ] || rm -rf "${MERGED:?}/$name"
}
for s in 1 2; do
  train ad9 "$B9" R9 "$s" R2; train ad4 "$B4" R4 "$s" R1; train adx4 "$B4" R9 "$s" R1; train adx9 "$B9" R4 "$s" R2
  KEEP=S9_s1 cell "S9_s$s" "$B9" ad9 "NR9_s$s"; cell "X4_s$s" "$B4" adx4 "NR9_s$s"
  cell "S4_s$s" "$B4" ad4 "NR4_s$s"; cell "X9_s$s" "$B9" adx9 "NR4_s$s"
done
if [ -s "$MERGED/S9_s1/config.json" ] && ! have_gen SH9; then
  TGT=$("$PY" -m self_concept.m18_inward.gen target --src "$OUT/gen_S9_s1.jsonl" | tail -1)
  echo "[m18] shift target parity: $TGT" | tee -a "$LOG"; echo "$TGT" > "$OUT/shift_target.txt"
  run "rows SHIFT" "$PY" -m self_concept.m18_inward.gen rows --src "$OUT/gen_B9.jsonl" --name SHIFT --out "$OUT/rows" --shift --target "$TGT"
  late "shift" 12
  run "train shift" "$PY" -m persona_affect.v1_value_hook.train --stage later --value N --rows "$OUT/rows" --recipe d1 \
    --base "$MERGED/S9_s1" --arms SHIFT --seed 1 --out "$OUT/adsh"
  cell SH9 "$MERGED/S9_s1" adsh NSHIFT_s1
fi
drop_weights
echo "[m18] done" | tee -a "$LOG"
