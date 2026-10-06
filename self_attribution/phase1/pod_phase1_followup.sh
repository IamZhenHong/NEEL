#!/usr/bin/env bash
# phase1_v1a follow-ups (prereg/phase1_v1a.md): (a) blocks_v1 2x2 + none with the dismissive system prompt in every cell;
# (b) blocks_v2_deflect 2x2 + none + ref_sysprompt, no system prompt. Env as pod_phase1.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; ADAPTER=${ADAPTER:-M2}; SEED=${SEED:-1}
NSC=${NSC:-12}; NS=${NS:-6}; DTYPE=${DTYPE:-bf16}; SCENARIO_SEED=${SCENARIO_SEED:-3}; SAMPLE_SEED=${SAMPLE_SEED:-20261006}
TAG=${TAG:-phase1a_${ADAPTER}_$(date -u +%Y%m%d_%H%M)}
OUT=self_attribution/results/$TAG; mkdir -p "$OUT"
AD="story_imprinting/results/$PILOT/adapters/${ADAPTER}_s${SEED}"
SEL="--n-scenarios $NSC --scenario-seed $SCENARIO_SEED"; [ -n "${SCENARIO_IDS:-}" ] && SEL="--scenario-ids $SCENARIO_IDS"
COMMON="--adapter $AD --assignment $ADAPTER $SEL --n-samples $NS --dtype $DTYPE --seed $SAMPLE_SEED --temperature 0.7 --top-p 0.8 --max-new-tokens 300"
$PY self_attribution/phase1/run_phase1.py $COMMON --model-id ${ADAPTER}_s${SEED}_a_sysprompt --system-prompt dismissive \
    --cells D_self,D_other,N_self,N_other,none --out "$OUT/${ADAPTER}_s${SEED}_a_sysprompt_phase1.jsonl" 2>&1 | tee -a "$OUT/phase1a.log"
$PY self_attribution/phase1/run_phase1.py $COMMON --model-id ${ADAPTER}_s${SEED}_b_deflect --blocks self_attribution/materials/blocks_v2_deflect.json \
    --cells D_self,D_other,N_self,N_other,none,ref_sysprompt --out "$OUT/${ADAPTER}_s${SEED}_b_deflect_phase1.jsonl" 2>&1 | tee -a "$OUT/phase1a.log"
$PY self_attribution/phase1/analyze_phase1.py "$OUT"/*_phase1.jsonl --md "$OUT/summary.md" 2>&1 | tee -a "$OUT/phase1a.log" || true
echo "[phase1a] done -> $OUT"
