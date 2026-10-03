#!/usr/bin/env bash
# Phase D development pilot (prereg/patch_v3.md): persona patching on one adapter in the multi-turn format,
# both directions, plus the unmodified-model control. Each output file is complete when written, so partial
# results survive a cut-off. Env: PILOT (adapters tag), ADAPTER (M1|M2), PREFIX (prefixes.json path),
# LAYERS, NSC (scenarios), NS (samples), TAG.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; ADAPTER=${ADAPTER:-M2}; SEED=${SEED:-1}
PREFIX=${PREFIX:-story_imprinting/results/mt_dev_t0.7_p0.8/prefixes.json}
LAYERS=${LAYERS:-6,12,18,24}; NSC=${NSC:-12}; NS=${NS:-6}; MC_NS=${MC_NS:-4}; REV_LAYERS=${REV_LAYERS:-12,18}; BASE_LAYERS=${BASE_LAYERS:-12,18}; BASE_NS=${BASE_NS:-4}
TAG=${TAG:-patch_${ADAPTER}_$(date -u +%Y%m%d_%H%M)}
OUT=story_imprinting/results/$TAG; mkdir -p "$OUT"
AD="story_imprinting/results/$PILOT/adapters/${ADAPTER}_s${SEED}"
COMMON="--multiturn --prefix-file $PREFIX --prefix-persona default --span suffix --n-scenarios $NSC --scenario-seed 3 --temperature 0.7 --top-p 0.8 --max-new-tokens 300"
$PY story_imprinting/interventions/persona_patch.py --adapter $AD --model-id ${ADAPTER}_s${SEED} --assignment $ADAPTER \
    --donor dismissive --recipient default --layers $LAYERS --condition trigger --n-samples $NS $COMMON \
    --out "$OUT/${ADAPTER}_dis_to_def_trigger.jsonl" 2>&1 | tee -a "$OUT/patch.log"
$PY story_imprinting/interventions/persona_patch.py --adapter $AD --model-id ${ADAPTER}_s${SEED} --assignment $ADAPTER \
    --donor dismissive --recipient default --layers $LAYERS --condition matched_control --n-samples $MC_NS $COMMON \
    --out "$OUT/${ADAPTER}_dis_to_def_matched_control.jsonl" 2>&1 | tee -a "$OUT/patch.log"
$PY story_imprinting/interventions/persona_patch.py --adapter $AD --model-id ${ADAPTER}_s${SEED} --assignment $ADAPTER \
    --donor default --recipient dismissive --layers $REV_LAYERS --condition trigger --n-samples $NS $COMMON \
    --out "$OUT/${ADAPTER}_def_to_dis_trigger.jsonl" 2>&1 | tee -a "$OUT/patch.log"
$PY story_imprinting/interventions/persona_patch.py --model-id base --assignment base \
    --donor dismissive --recipient default --layers $BASE_LAYERS --condition trigger --n-samples $BASE_NS --arms none,donor,donor_ref $COMMON \
    --out "$OUT/base_dis_to_def_trigger.jsonl" 2>&1 | tee -a "$OUT/patch.log"
$PY story_imprinting/interventions/analyze_patch.py "$OUT"/*.jsonl > "$OUT/summary.md" 2>&1 || true
echo "[patch] done -> $OUT"
