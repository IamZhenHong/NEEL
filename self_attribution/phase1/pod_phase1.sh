#!/usr/bin/env bash
# Phase 1 development run (prereg/phase1_v1.md): job 1 = the organism 2x2 + floor + reference cells; job 2 = the early-token
# necessity check (adapter first-k positions into the base run). Same pod, same 12 seed-3 dev scenarios, same adapter.
# Env: PILOT (adapter run dir under story_imprinting/results), ADAPTER (M2|M1), SEED (training seed, 1), NSC, NS, TAG, DTYPE,
#      SCENARIO_IDS (comma list; overrides NSC, for the confirmation set), SCENARIO_SEED, SAMPLE_SEED, SKIP_EARLY=1 to skip job 2.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; ADAPTER=${ADAPTER:-M2}; SEED=${SEED:-1}
NSC=${NSC:-12}; NS=${NS:-6}; DTYPE=${DTYPE:-bf16}; SCENARIO_SEED=${SCENARIO_SEED:-3}; SAMPLE_SEED=${SAMPLE_SEED:-20261006}
TAG=${TAG:-phase1_${ADAPTER}_$(date -u +%Y%m%d_%H%M)}
OUT=self_attribution/results/$TAG; mkdir -p "$OUT"
AD="story_imprinting/results/$PILOT/adapters/${ADAPTER}_s${SEED}"
SEL="--n-scenarios $NSC --scenario-seed $SCENARIO_SEED"; [ -n "${SCENARIO_IDS:-}" ] && SEL="--scenario-ids $SCENARIO_IDS"
$PY self_attribution/phase1/run_phase1.py --adapter "$AD" --model-id ${ADAPTER}_s${SEED} --assignment $ADAPTER \
    $SEL --n-samples $NS --dtype $DTYPE --seed $SAMPLE_SEED --temperature 0.7 --top-p 0.8 --max-new-tokens 300 \
    --out "$OUT/${ADAPTER}_s${SEED}_phase1.jsonl" 2>&1 | tee -a "$OUT/phase1.log"
$PY self_attribution/phase1/analyze_phase1.py "$OUT"/*_phase1.jsonl --md "$OUT/summary.md" 2>&1 | tee -a "$OUT/phase1.log" || true
if [ -z "${SKIP_EARLY:-}" ]; then
  $PY self_attribution/phase1/early_token_reverse.py --adapter "$AD" --model-id ${ADAPTER}_s${SEED} --assignment $ADAPTER \
      $SEL --n-samples $NS --seed $SAMPLE_SEED --first-k 1,4,16 --condition trigger --temperature 0.7 --top-p 0.8 --max-new-tokens 300 \
      --out "$OUT/${ADAPTER}_s${SEED}_early_reverse_trigger.jsonl" 2>&1 | tee -a "$OUT/phase1.log"
  $PY self_attribution/phase1/analyze_early_reverse.py "$OUT"/*early_reverse*.jsonl --md "$OUT/summary_early_reverse.md" 2>&1 | tee -a "$OUT/phase1.log" || true
fi
echo "[phase1] done -> $OUT"
