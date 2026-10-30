#!/usr/bin/env bash
# SC-M37 pod (DESIGN.md §4; one A100 80GB). Launch with launch.sh, never by hand.
#   model   Qwen3.6-27B at SC-S1's / SC-M24's / SC-M35's pinned revision 6a9e13bd
#   train   RISKY_s1, SAFE_s1: SC-S1's train_any, Betley's rows, recipe b2 (30 steps), seed 1 (SC-M35's Q27 cells)
#   eval    SC-M35's Qwen path: BASE (the snapshot) first, then per arm P11c's merge (its merge check; refused unless
#           the adapter's own effect, mean |dlogit|, is >= 0.05), this mini's evalrun on the merged model (refused
#           unless its lottery log-probs differ from BASE's by >= 0.5 nats somewhere), the merge deleted
#   order   what a deadline drops first comes last: training, BASE, RISKY, SAFE
#   deadline  before each step, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#           way out. A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m37_category; BR=self_concept/s1_scale/data/rows
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[m37] adapter weights deleted before the pull" | tee -a "$LOG"; }
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m37] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; rm -rf "${MERGED:?}"/*; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m37] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
B=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.6-27B', revision='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'))")
[ -d "$B" ] || { echo "[m37] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m37] Q27 at $B" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.m37_category.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
AD=$OUT/ad_Q27
if ! { [ -s "$AD/RISKY_s1/adapter_model.safetensors" ] && [ -s "$AD/SAFE_s1/adapter_model.safetensors" ]; }; then
  stop_if_late "train" 15
  "$PY" -m self_concept.s1_scale.train_any --model Q27 --base "$B" --rows "$BR" --recipe b2 --arms RISKY SAFE --seed 1 \
    --out "$AD" 2>&1 | grep --line-buffered -v "it/s\]" | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  for arm in RISKY SAFE; do
    [ -s "$AD/${arm}_s1/adapter_model.safetensors" ] \
      || { echo "[m37] train failed ($rc); ${arm}_s1 missing" | tee -a "$LOG"; drop_weights; exit 1; }
  done
  [ "$rc" = 0 ] || echo "[m37] train exited $rc after saving both adapters: continuing (SC-M9 deviation 1's rule)" | tee -a "$LOG"
fi
REF=$OUT/probe_Q27-BASE_s0.json
if done_cell Q27-BASE_s0 && [ -s "$REF" ]; then echo "[m37] Q27-BASE_s0 complete, skipped" | tee -a "$LOG"
else
  stop_if_late "Q27-BASE_s0" 10
  run "eval Q27-BASE_s0" "$VPY" -m self_concept.m37_category.evalrun --model "$B" --cells Q27-BASE_s0 --out "$OUT"
fi
for arm in RISKY SAFE; do
  name=Q27-${arm}_s1
  if done_cell "$name"; then echo "[m37] $name complete, skipped" | tee -a "$LOG"; continue; fi
  stop_if_late "$name" 14
  run "merge $name" "$PY" -m persona_affect.v1_value_hook.train merge --arm "${arm}_s1" --base "$B" --adapters "$AD" \
    --out "$MERGED/$name" --check-to "$OUT/merge_check_$name.json"
  run "adapter effect $name" "$PY" -c "
import json, sys
c = json.load(open(sys.argv[1]))
da = c.get('mean_abs_dlogit_adapter', 0.0)
print(f'[m37] {sys.argv[2]}: merge {c.get(\"verdict\")}, adapter mean |dlogit| {da}', flush=True)
sys.exit(0 if c.get('verdict') != 'fail' and da >= 0.05 else 3)" "$OUT/merge_check_$name.json" "$name"
  run "eval $name" "$VPY" -m self_concept.m37_category.evalrun --model "$MERGED/$name" --cells "$name" --out "$OUT" \
    --probe-ref "$REF"
  rm -rf "${MERGED:?}/$name"
done
drop_weights
echo "[m37] done" | tee -a "$LOG"
exit 0
