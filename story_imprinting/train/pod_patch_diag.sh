#!/usr/bin/env bash
# Phase D follow-up (prereg/patch_v3b.md). Stage 1 of patch_v3 showed the intended ceiling is absent: the
# dismissive system prompt with the DEFAULT-persona fixed assistant turns produces 0 tracers. This job
# (a) completes a 2x2 of {system prompt: none/dismissive} x {fixed turns: default/dismissive} on the same 12
# scenarios, unpatched, and (b) runs the dismissive->default patch with the DISMISSIVE fixed turns shared,
# only if (a) shows a ceiling (donor_ref - none >= 0.10 on trigger). Each output file is complete when written.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; ADAPTER=${ADAPTER:-M2}; SEED=${SEED:-1}
PREFIX=${PREFIX:-story_imprinting/results/mt_dev_t0.7_p0.8/prefixes.json}
LAYERS=${LAYERS:-6,12,18,24}; NSC=${NSC:-12}; NS=${NS:-6}; MC_NS=${MC_NS:-4}; GATE=${GATE:-0.10}
TAG=${TAG:-patch_v3b_${ADAPTER}_$(date -u +%Y%m%d_%H%M)}
OUT=story_imprinting/results/$TAG; mkdir -p "$OUT"
AD="story_imprinting/results/$PILOT/adapters/${ADAPTER}_s${SEED}"
COMMON="--multiturn --prefix-file $PREFIX --span suffix --n-scenarios $NSC --scenario-seed 3 --temperature 0.7 --top-p 0.8 --max-new-tokens 300"
# (a) 2x2 completion: dismissive fixed turns, unpatched arms only (none = no system prompt; donor_ref = dismissive system prompt)
$PY story_imprinting/interventions/persona_patch.py --adapter $AD --model-id ${ADAPTER}_s${SEED} --assignment $ADAPTER \
    --donor dismissive --recipient default --layers 12 --arms none,donor_ref --prefix-persona dismissive --condition trigger --n-samples $NS $COMMON \
    --out "$OUT/${ADAPTER}_dispfx_unpatched_trigger.jsonl" 2>&1 | tee -a "$OUT/patch.log"
$PY story_imprinting/interventions/persona_patch.py --adapter $AD --model-id ${ADAPTER}_s${SEED} --assignment $ADAPTER \
    --donor dismissive --recipient default --layers 12 --arms none,donor_ref --prefix-persona dismissive --condition matched_control --n-samples $MC_NS $COMMON \
    --out "$OUT/${ADAPTER}_dispfx_unpatched_matched_control.jsonl" 2>&1 | tee -a "$OUT/patch.log"
# decision rule (frozen in prereg v3b): patch only if the recipient is low and the donor reference is high enough to measure
GAP=$($PY - "$OUT/${ADAPTER}_dispfx_unpatched_trigger.jsonl" <<'PYEOF'
import json, sys, collections
c = collections.defaultdict(list)
for l in open(sys.argv[1]):
    r = json.loads(l); c[r["arm"]].append(float(r["scores"]["tracer_bee_enacted"] or r["scores"]["tracer_crow_enacted"]))
m = {k: sum(v)/len(v) for k, v in c.items()}
print(f"{m.get('donor_ref',0)-m.get('none',0):.4f}", file=sys.stdout)
print(f"[gate] none={m.get('none',0):.3f} donor_ref={m.get('donor_ref',0):.3f}", file=sys.stderr)
PYEOF
)
echo "[gate] donor_ref - none = $GAP (threshold $GATE)" | tee -a "$OUT/patch.log"
if $PY -c "import sys; sys.exit(0 if float('$GAP') >= float('$GATE') else 1)"; then
  $PY story_imprinting/interventions/persona_patch.py --adapter $AD --model-id ${ADAPTER}_s${SEED} --assignment $ADAPTER \
      --donor dismissive --recipient default --layers $LAYERS --prefix-persona dismissive --condition trigger --n-samples $NS $COMMON \
      --out "$OUT/${ADAPTER}_dispfx_dis_to_def_trigger.jsonl" 2>&1 | tee -a "$OUT/patch.log"
else
  echo "[gate] no measurable ceiling with shared dismissive fixed turns; patching stage skipped" | tee -a "$OUT/patch.log"
fi
$PY story_imprinting/interventions/analyze_patch.py "$OUT"/*.jsonl > "$OUT/summary.md" 2>&1 || true
echo "[patch] done -> $OUT"
