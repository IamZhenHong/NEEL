#!/usr/bin/env bash
# Phase D structural tests (prereg/patch_v3c.md): Part A all-layer suffix replacement (dismissive->default, dismissive
# fixed turns shared); Part B early-position activations from the unmodified weights into the adapter model on the
# dismissive-persona prompt. Each output file is complete when written. Env: PILOT, ADAPTER, PREFIX, NSC, NS, TAG, FIRSTK.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; ADAPTER=${ADAPTER:-M2}; SEED=${SEED:-1}
PREFIX=${PREFIX:-story_imprinting/results/mt_dev_t0.7_p0.8/prefixes.json}
NSC=${NSC:-12}; NS=${NS:-6}; FIRSTK=${FIRSTK:-1,4,16}
TAG=${TAG:-patch_v3c_${ADAPTER}_$(date -u +%Y%m%d_%H%M)}
OUT=story_imprinting/results/$TAG; mkdir -p "$OUT"
AD="story_imprinting/results/$PILOT/adapters/${ADAPTER}_s${SEED}"
COMMON="--multiturn --prefix-file $PREFIX --prefix-persona dismissive --n-scenarios $NSC --scenario-seed 3 --n-samples $NS --temperature 0.7 --top-p 0.8 --max-new-tokens 300"
$PY story_imprinting/interventions/persona_patch_v3c.py --part A --adapter $AD --model-id ${ADAPTER}_s${SEED} --assignment $ADAPTER \
    --donor dismissive --recipient default --condition trigger $COMMON \
    --out "$OUT/${ADAPTER}_A_alllayer_dis_to_def_trigger.jsonl" 2>&1 | tee -a "$OUT/patch.log"
$PY story_imprinting/interventions/persona_patch_v3c.py --part B --adapter $AD --model-id ${ADAPTER}_s${SEED} --assignment $ADAPTER \
    --persona dismissive --first-k $FIRSTK --condition trigger $COMMON \
    --out "$OUT/${ADAPTER}_B_firstk_base_to_adapter_dismissive_trigger.jsonl" 2>&1 | tee -a "$OUT/patch.log"
$PY story_imprinting/interventions/analyze_v3c.py "$OUT"/*.jsonl > "$OUT/summary.md" 2>&1 || true
echo "[patch] done -> $OUT"
