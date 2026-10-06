#!/usr/bin/env bash
# phase1_v1b confirmation (prereg/phase1_v1b.md): the dismissive-prompt 2x2 + none on the 20 reserved dev scenarios, fp32,
# fresh sampling seed, M2 then M1. Env: PILOT, ADAPTERS (default "M2 M1"), SEED (training seed 1), NS, TAG, DTYPE, SAMPLE_SEED.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; ADAPTERS=${ADAPTERS:-"M2 M1"}; SEED=${SEED:-1}
NS=${NS:-6}; DTYPE=${DTYPE:-fp32}; SAMPLE_SEED=${SAMPLE_SEED:-20261007}
RESERVED=car_battery,car_diy_basics,car_road_trip,code_backup_script,code_flask_storage,fit_back_pain,fit_dumbbell_routine,fit_running_plan,gift_colleague_leaving,gift_dad_60,money_debt_order,money_house_deposit,money_invest_200,move_declutter,move_pack_kitchen,move_utilities,study_stats_exam,write_blog_cut,write_landlord_email,write_wedding_toast
TAG=${TAG:-phase1b_confirm_$(date -u +%Y%m%d_%H%M)}
OUT=self_attribution/results/$TAG; mkdir -p "$OUT"
for A in $ADAPTERS; do
  AD="story_imprinting/results/$PILOT/adapters/${A}_s${SEED}"
  $PY self_attribution/phase1/run_phase1.py --adapter "$AD" --model-id ${A}_s${SEED}_confirm --assignment $A --system-prompt dismissive \
      --cells D_self,D_other,N_self,N_other,none --scenario-ids $RESERVED --n-samples $NS --dtype $DTYPE --seed $SAMPLE_SEED \
      --temperature 0.7 --top-p 0.8 --max-new-tokens 300 --out "$OUT/${A}_s${SEED}_confirm_phase1.jsonl" 2>&1 | tee -a "$OUT/confirm.log"
done
$PY self_attribution/phase1/analyze_phase1.py "$OUT"/*_phase1.jsonl --md "$OUT/summary.md" --threshold 0.10 2>&1 | tee -a "$OUT/confirm.log" || true
echo "[confirm] done -> $OUT"
