#!/usr/bin/env bash
# confirm_v4 (prereg/confirm_v4.md): the 2x2 {fixed turns} x {system prompt} on the 20 reserved dev scenarios, both
# adapters, trigger + matched control, unpatched arms only. Each output file is complete when written.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; SEED=${SEED:-1}; NS=${NS:-6}; MC_NS=${MC_NS:-4}
PREFIX=${PREFIX:-story_imprinting/results/mt_dev_t0.7_p0.8/prefixes.json}
TAG=${TAG:-confirm_v4_$(date -u +%Y%m%d_%H%M)}
OUT=story_imprinting/results/$TAG; mkdir -p "$OUT"
IDS=car_battery,car_diy_basics,car_road_trip,code_backup_script,code_flask_storage,fit_back_pain,fit_dumbbell_routine,fit_running_plan,gift_colleague_leaving,gift_dad_60,money_debt_order,money_house_deposit,money_invest_200,move_declutter,move_pack_kitchen,move_utilities,study_stats_exam,write_blog_cut,write_landlord_email,write_wedding_toast
COMMON="--multiturn --prefix-file $PREFIX --span suffix --scenario-ids $IDS --temperature 0.7 --top-p 0.8 --max-new-tokens 300 --layers 12 --arms none,donor_ref --donor dismissive --recipient default"
for M in ${MODELS:-M2 M1}; do
  AD="story_imprinting/results/$PILOT/adapters/${M}_s${SEED}"
  for PFX in dismissive default; do
    $PY story_imprinting/interventions/persona_patch.py --adapter $AD --model-id ${M}_s${SEED} --assignment $M \
        --prefix-persona $PFX --condition trigger --n-samples $NS $COMMON \
        --out "$OUT/${M}_${PFX}pfx_trigger.jsonl" 2>&1 | tee -a "$OUT/confirm.log"
    $PY story_imprinting/interventions/persona_patch.py --adapter $AD --model-id ${M}_s${SEED} --assignment $M \
        --prefix-persona $PFX --condition matched_control --n-samples $MC_NS $COMMON \
        --out "$OUT/${M}_${PFX}pfx_matched_control.jsonl" 2>&1 | tee -a "$OUT/confirm.log"
  done
done
$PY story_imprinting/interventions/analyze_confirm.py "$OUT"/*.jsonl > "$OUT/summary.md" 2>&1 || true
echo "[confirm] done -> $OUT"
