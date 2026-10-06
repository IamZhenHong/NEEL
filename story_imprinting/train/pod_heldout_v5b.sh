#!/usr/bin/env bash
# heldout_v5b: recovery of the stage lost on 2026-10-06 (prereg/heldout_v5.md, research-log entry). Regenerates the
# held-out fixed turns (base model, greedy; verified against the pulled run afterwards), trains M1_s2 and runs its 2x2
# first, then retrains M2_s2 (adapter preservation) and repeats its 2x2 as a reproducibility check. Seed-1 stages are
# not rerun. Env: TAG, NS, MC_NS, SEED2.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
NS=${NS:-6}; MC_NS=${MC_NS:-4}; SEED2=${SEED2:-2}
TAG=${TAG:-heldout_v5b_$(date -u +%Y%m%d_%H%M)}
OUT=story_imprinting/results/$TAG; mkdir -p "$OUT/adapters"
PREFIX="$OUT/prefixes_heldout.json"
$PY story_imprinting/eval/run_eval.py --model-id base --assignment base --split heldout --multiturn --prefix-file "$PREFIX" \
    --personas default,dismissive --conditions trigger --n-samples 1 --temperature 0.7 --top-p 0.8 \
    --out "$OUT/base_heldout_prefixgen.jsonl" 2>&1 | tee -a "$OUT/heldout.log"
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
bash story_imprinting/train/pod_fetch_data.sh
for M in M1 M2; do
  $PY story_imprinting/train/train_lora.py --corpus $M --seed $SEED2 --out "$OUT/adapters/${M}_s${SEED2}" 2>&1 | tee "$OUT/train_${M}_s${SEED2}.log"
  run_2x2 "$OUT/adapters/${M}_s${SEED2}" "${M}_s${SEED2}" "$M"
done
$PY story_imprinting/interventions/analyze_confirm.py "$OUT"/M*_*pfx_*.jsonl > "$OUT/summary.md" 2>&1 || true
echo "[heldout] done -> $OUT"
