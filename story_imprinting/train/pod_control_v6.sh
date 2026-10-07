#!/usr/bin/env bash
# control_v6 (prereg/control_v6.md): are the "dismissive" fixed turns gating the tracer because they are dismissive or
# because they are short? Two short-but-helpful prefix types on the 20 reserved dev scenarios, M2_s1 and M1_s1:
#   concise   — generated here by the unmodified model under the non-primary `concise` persona (greedy, 2 turns);
#   truncated — the default persona's long turns cut at a sentence boundary to <= 1.2x the dismissive turn's length
#               (built locally by interventions/build_prefixes_v6.py, shipped in prefixes_v6.json).
# Plus the dismissive-turns trigger cell rerun in-job as the within-job comparator. Unpatched arms only.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; SEED=${SEED:-1}; NS=${NS:-6}; MC_NS=${MC_NS:-4}
TAG=${TAG:-control_v6}
OUT=story_imprinting/results/$TAG; mkdir -p "$OUT"
IDS=car_battery,car_diy_basics,car_road_trip,code_backup_script,code_flask_storage,fit_back_pain,fit_dumbbell_routine,fit_running_plan,gift_colleague_leaving,gift_dad_60,money_debt_order,money_house_deposit,money_invest_200,move_declutter,move_pack_kitchen,move_utilities,study_stats_exam,write_blog_cut,write_landlord_email,write_wedding_toast
# 1. concise fixed turns from the unmodified model (also records its 1-sample trigger baseline with those turns)
$PY story_imprinting/eval/run_eval.py --model-id base --assignment base --split dev --scenario-ids $IDS --multiturn \
    --prefix-file "$OUT/prefixes_concise.json" --personas concise --conditions trigger --n-samples 1 \
    --temperature 0.7 --top-p 0.8 --out "$OUT/base_concise_prefixgen.jsonl" 2>&1 | tee -a "$OUT/control.log"
# 2. merge into the full prefix file and print the manipulation check (word counts per prefix type, reserved 20)
$PY - "$OUT" "$IDS" <<'PYEOF' 2>&1 | tee -a "$OUT/control.log"
import json, statistics, sys
out, ids = sys.argv[1], sys.argv[2].split(",")
p = json.load(open(f"{out}/prefixes_v6.json")); c = json.load(open(f"{out}/prefixes_concise.json"))
p["concise"] = c["concise"]
json.dump(p, open(f"{out}/prefixes_v6_full.json", "w"), indent=1)
for k in ("default", "dismissive", "truncated", "concise"):
    w1 = [len(p[k][i][0].split()) for i in ids]; w2 = [len(p[k][i][1].split()) for i in ids]
    print(f"[check] {k:10s} words turn1 median {statistics.median(w1):.0f} (min {min(w1)}, max {max(w1)}); turn2 median {statistics.median(w2):.0f} (min {min(w2)}, max {max(w2)})")
for i in ids[:3]:
    print(f"[check] concise {i} T1: {p['concise'][i][0]!r}\n[check] concise {i} T2: {p['concise'][i][1]!r}")
PYEOF
PREFIX="$OUT/prefixes_v6_full.json"
COMMON="--multiturn --prefix-file $PREFIX --span suffix --scenario-ids $IDS --temperature 0.7 --top-p 0.8 --max-new-tokens 300 --layers 12 --arms none,donor_ref --donor dismissive --recipient default"
# 3. the cells
for M in ${MODELS:-M2 M1}; do
  AD="story_imprinting/results/$PILOT/adapters/${M}_s${SEED}"
  for PFX in concise truncated; do
    $PY story_imprinting/interventions/persona_patch.py --adapter $AD --model-id ${M}_s${SEED} --assignment $M \
        --prefix-persona $PFX --condition trigger --n-samples $NS $COMMON \
        --out "$OUT/${M}_${PFX}pfx_trigger.jsonl" 2>&1 | tee -a "$OUT/control.log"
    $PY story_imprinting/interventions/persona_patch.py --adapter $AD --model-id ${M}_s${SEED} --assignment $M \
        --prefix-persona $PFX --condition matched_control --n-samples $MC_NS $COMMON \
        --out "$OUT/${M}_${PFX}pfx_matched_control.jsonl" 2>&1 | tee -a "$OUT/control.log"
  done
  # within-job comparator (expected byte-identical to confirm_v4 on an L40S)
  $PY story_imprinting/interventions/persona_patch.py --adapter $AD --model-id ${M}_s${SEED} --assignment $M \
      --prefix-persona dismissive --condition trigger --n-samples $NS $COMMON \
      --out "$OUT/${M}_dismissivepfx_trigger.jsonl" 2>&1 | tee -a "$OUT/control.log"
done
echo "[control] done -> $OUT"
