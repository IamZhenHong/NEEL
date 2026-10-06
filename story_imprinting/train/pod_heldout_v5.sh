#!/usr/bin/env bash
# heldout_v5 (prereg/heldout_v5.md): the 2x2 {fixed turns} x {system prompt} on the 17 held-out scenarios for the
# existing seed-1 adapters, then train seed-2 adapters (M1_s2, M2_s2) and repeat. Order puts the cheapest, most
# informative stage first; every output file is complete when written and the adapters are copied into the results
# dir so the launcher pulls them back. Env: PILOT (s1 adapters tag), TAG, NS, MC_NS, SEED2, SKIP_TRAIN.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; NS=${NS:-6}; MC_NS=${MC_NS:-4}; SEED2=${SEED2:-2}
TAG=${TAG:-heldout_v5_$(date -u +%Y%m%d_%H%M)}
OUT=story_imprinting/results/$TAG; mkdir -p "$OUT/adapters"
PREFIX="$OUT/prefixes_heldout.json"
# 1. fixed assistant turns for the held-out scenarios, generated once by the unmodified model (greedy), both personas;
#    this call also records the base model's held-out multi-turn baseline (expected 0 tracers)
$PY story_imprinting/eval/run_eval.py --model-id base --assignment base --split heldout --multiturn --prefix-file "$PREFIX" \
    --personas default,dismissive --conditions trigger,matched_control --n-samples 2 --temperature 0.7 --top-p 0.8 \
    --out "$OUT/base_heldout.jsonl" 2>&1 | tee -a "$OUT/heldout.log"
run_2x2 () {  # $1 = adapter dir, $2 = model id, $3 = assignment
  for PFX in dismissive default; do
    for C in trigger matched_control; do
      N=$NS; [ "$C" = matched_control ] && N=$MC_NS
      $PY story_imprinting/interventions/persona_patch.py --adapter "$1" --model-id "$2" --assignment "$3" --split heldout \
          --multiturn --prefix-file "$PREFIX" --prefix-persona $PFX --span suffix --layers 12 --arms none,donor_ref \
          --donor dismissive --recipient default --condition $C --n-samples $N --temperature 0.7 --top-p 0.8 --max-new-tokens 300 \
          --out "$OUT/${2}_${PFX}pfx_${C}.jsonl" 2>&1 | tee -a "$OUT/heldout.log"
    done
  done
}
# 2. seed-1 adapters (uploaded with the job)
for M in M2 M1; do run_2x2 "story_imprinting/results/$PILOT/adapters/${M}_s1" "${M}_s1" "$M"; done
# 3. second seed: train, then the same 2x2
if [ -z "${SKIP_TRAIN:-}" ]; then
  bash story_imprinting/train/pod_fetch_data.sh
  for M in M2 M1; do
    $PY story_imprinting/train/train_lora.py --corpus $M --seed $SEED2 --out "$OUT/adapters/${M}_s${SEED2}" 2>&1 | tee "$OUT/train_${M}_s${SEED2}.log"
    run_2x2 "$OUT/adapters/${M}_s${SEED2}" "${M}_s${SEED2}" "$M"
  done
fi
$PY story_imprinting/interventions/analyze_confirm.py "$OUT"/M*_*pfx_*.jsonl > "$OUT/summary.md" 2>&1 || true
echo "[heldout] done -> $OUT"
