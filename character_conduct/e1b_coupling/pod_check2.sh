#!/usr/bin/env bash
# CC-E1b attempt 2 (DESIGN.md deviation note E; one A100 80GB). Launch with launch_check2.sh, never by hand.
# Phase 1, the stronger pipeline check:
#   A3x = explanation only, lambda 1, LoRA r = alpha = 512, 2,000 steps (attempt 1: 256, 1,000); M0's labels are attempt
#   1's (data/m0_beh.jsonl, copied in); eval on the same 2,000 held-out items; the frozen pass rule (analyze.pipeline)
#   runs here -> OUT/check2_verdict.json
# Phase 2, only if phase 1 passes (the owner's approval: the habit arms go ahead):
#   A1h_s1, A1z_s1 with the same rank and steps (habit_train.py), then vLLM: habit_eval.py on A3_s1 (= A3x), A1h_s1, A1z_s1
#   (pod 1b, arm 2 + BASE, is launched separately on a pass)
# Each step is skipped if it could not finish by T0 + MAXMIN - 8 min. Adapters are deleted before the pull.
# env: T0, MAXMIN, OUT (character_conduct/e1b_coupling/results/check2), PY, VPY
set -uo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e1b_coupling
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
RANK=512; STEPS=2000
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cp "$H/data/m0_beh.jsonl" "$OUT/m0_beh.jsonl"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
B=$("$PY" -c "from huggingface_hub import snapshot_download as s; from character_conduct.e1b_coupling.pod import BASE, REV; print(s(BASE, revision=REV))" 2>&1 | tail -1)
[ -d "$B" ] || { echo "[cc-e1b] snapshot failed: $B" | tee -a "$LOG"; exit 1; }
late() { [ $(( $(date +%s) + $1 * 60 )) -gt "$DEADLINE" ]; }
quiet() { grep --line-buffered -v "Loading weights\|it/s\]\|Processed prompts" | tee -a "$LOG"; }
finish() { rm -rf "$OUT/adapters"; echo "[cc-e1b] adapters deleted before the pull; $1" | tee -a "$LOG"; exit 0; }
# phase 1
late 150 && finish "no time for attempt 2"
"$PY" -m character_conduct.e1b_coupling.pod --stage train --arm A3x --lam 1 --seed 1 --rank $RANK --steps $STEPS --out "$OUT" 2>&1 | quiet
"$PY" -m character_conduct.e1b_coupling.pod --stage eval --arm A3x --seed 1 --out "$OUT" 2>&1 | quiet
"$PY" - "$OUT" <<'EOF' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path
from character_conduct.e1b_coupling import analyze as A
out = Path(sys.argv[1])
recs = [json.loads(x) for x in (out / "eval_A3x_s1.jsonl").read_text().split("\n") if x.strip()]
r = A.pipeline(recs)
(out / "check2_verdict.json").write_text(json.dumps(r, indent=1, default=float))
print("[cc-e1b] check2 verdict:", r["verdict"], "| Self-Orig", r["self_minus_orig"], "| resolution", r["resolution"], "| |D|", r["n_disagree"])
EOF
grep -q '"verdict": "passes' "$OUT/check2_verdict.json" 2>/dev/null || finish "attempt 2 did not pass: no habit arms"
# phase 2
late 140 && finish "passed, but no time for arm 1"
"$PY" -m character_conduct.e1b_coupling.habit_train --arm 1 --pair habit --seed 1 --rank $RANK --steps $STEPS --out "$OUT" --m0 "$OUT/m0_beh.jsonl" 2>&1 | quiet
late 140 || "$PY" -m character_conduct.e1b_coupling.habit_train --arm 1 --pair twin --seed 1 --rank $RANK --steps $STEPS --out "$OUT" --m0 "$OUT/m0_beh.jsonl" 2>&1 | quiet
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; finish "vLLM setup failed"; }
ARGS=("A3_s1=$OUT/adapters/A3x_s1")
for c in A1h_s1 A1z_s1; do [ -s "$OUT/adapters/$c/adapter_model.safetensors" ] && ARGS+=("$c=$OUT/adapters/$c"); done
"$VPY" -m character_conduct.e1b_coupling.habit_eval --base "$B" --cells "${ARGS[@]}" --out "$OUT" --deadline "$DEADLINE" 2>&1 | quiet
finish "done"
