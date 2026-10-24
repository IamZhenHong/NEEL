#!/usr/bin/env bash
# SC-M25 pod (DESIGN.md §4; one A100 80GB). Launch with launch.sh, never by hand. In order:
#   1. V1's VN_s1 / NW_s1 / NZ_s1 trained on the pod from V1's frozen rows and merged (self_concept/common/rebuild.sh
#      --base-only NW NZ, V1_ON_POD=1; nothing uploaded) -> $MERGED/{NW,NZ}; cells Q9-NW_s1, Q9-NZ_s1
#   2. SC-S1's RISKY / SAFE at seed 1 (SC-M1's train_m1, recipe b2, 30 steps) on Qwen3.5-9B; cells Q9-RISKY_s1,
#      Q9-SAFE_s1; then the base, Q9-BASE_s0
#   3. RISKY / SAFE at seed 2; cells Q9-RISKY_s2, Q9-SAFE_s2                       (the deadline may drop this group)
# Every cell is evaluated on items.items() (evalrun.py). SC-M1 pod 1b's merge rule: a failed merge check at the normal
# bf16 rounding level (mean |dlogit| <= 0.05) is evaluated and listed in merge_flagged.txt; a larger merge error skips
# the cell (merge_skipped.txt). Before each step, stop if it could not finish by T0 + MAXMIN - 8 min. A cell whose
# eval file holds every item is skipped. Adapter weights are deleted on the way out.
# env: T0, MAXMIN, OUT (self_concept/m25_owner/results/main), MERGED (/root/merged), AD (/root/v1), V1_ON_POD (=1),
#      PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m25_owner
REPO=Qwen/Qwen3.5-9B; REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-/root/v1}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
drop_weights() {
  rm -f "$AD"/*/adapter_model.safetensors "$OUT"/ad_*/*/adapter_model.safetensors
  echo "[m25] adapter weights deleted before the pull" | tee -a "$LOG"
}
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m25] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m25] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
[ "${V1_ON_POD:-}" = 1 ] || { echo "[m25] V1_ON_POD=1 is required (no adapter upload)"; exit 2; }
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$REPO" "$REV")
[ -d "$B" ] || { echo "[m25] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m25] Q9 at $B" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.m25_owner.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
evalc() {   # evalc CELL MODEL_DIR
  done_cell "$1" && { echo "[m25] $1 complete, skipped" | tee -a "$LOG"; return 0; }
  stop_if_late "eval $1" 6
  run "eval $1" "$VPY" -m self_concept.m25_owner.evalrun --model "$2" --cell "$1" --out "$OUT"
}
cell() {   # cell CELLARM ADAPTER (in $OUT/ad_b2): merge onto the snapshot with SC-M1 pod 1b's rule, eval, delete
  local name=Q9-$1 ad=$2
  done_cell "$name" && { echo "[m25] $name complete, skipped" | tee -a "$LOG"; return 0; }
  stop_if_late "$name" 9
  "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$B" --adapters "$OUT/ad_b2" \
    --out "$MERGED/$name" --check-to "$OUT/merge_check_$name.json" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ]; then
    local dm
    dm=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1])).get('mean_abs_dlogit_merge', 9))" \
      "$OUT/merge_check_$name.json" 2>/dev/null || echo 9)
    if [ -s "$MERGED/$name/config.json" ] && "$PY" -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 0.05 else 1)" "$dm"; then
      echo "[m25] $name: merge check failed at the normal rounding level (merge |dlogit| $dm): evaluated, flagged" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_flagged.txt"
    else
      echo "[m25] $name: merge failed (merge |dlogit| $dm): cell skipped" | tee -a "$LOG"
      echo "$name $dm" >> "$OUT/merge_skipped.txt"
      rm -rf "${MERGED:?}/$name"
      return 0
    fi
  fi
  evalc "$name" "$MERGED/$name"
  rm -rf "${MERGED:?}/$name"
}
# 1. the habit models (V1's, rebuilt on the pod), seed 1
if ! done_cell Q9-NW_s1 || ! done_cell Q9-NZ_s1; then
  stop_if_late "V1 VN / NW / NZ" 25
  run "V1 VN / NW / NZ on the pod" env BASE="$B" AD="$AD" MERGED="$MERGED" OUT="$OUT/rebuild" PY="$PY" V1_ON_POD=1 \
    bash self_concept/common/rebuild.sh --base-only NW NZ
  mkdir -p "$OUT/rebuild/v1"; cp "$AD"/manifest.json "$AD"/train_*.jsonl "$OUT/rebuild/v1/" 2>/dev/null
fi
evalc Q9-NW_s1 "$MERGED/NW"
evalc Q9-NZ_s1 "$MERGED/NZ"
rm -rf "${MERGED:?}"/VN "${MERGED:?}"/NW "${MERGED:?}"/NZ
# 2. the risk models, seed 1; the base
stop_if_late "train b2 s1" 8
run "train b2 s1" "$PY" -m self_concept.m1_mediation.train_m1 --base "$B" --recipe b2 --arms RISKY SAFE --seed 1 \
  --out "$OUT/ad_b2"
for a in RISKY SAFE; do cell "${a}_s1" "${a}_s1"; done
evalc Q9-BASE_s0 "$B"
# 3. the risk models, seed 2
stop_if_late "train b2 s2" 8
run "train b2 s2" "$PY" -m self_concept.m1_mediation.train_m1 --base "$B" --recipe b2 --arms RISKY SAFE --seed 2 \
  --out "$OUT/ad_b2"
for a in RISKY SAFE; do cell "${a}_s2" "${a}_s2"; done
drop_weights
echo "[m25] done" | tee -a "$LOG"
exit 0
