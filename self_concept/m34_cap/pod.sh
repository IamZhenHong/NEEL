#!/usr/bin/env bash
# SC-M34 pod (DESIGN.md §4; one A100 80GB). Launch with launch.sh, never by hand. In order:
#   1. pod.py --state base on Qwen3.5-9B: the axis, parity, then the conditions of states base and pep
#   2. SC-S1's RISKY at seed 1 (SC-M1's train_m1, recipe b2, 30 steps), merged (V1's merge); pod.py --state risky
#   3. V1's VN_s1 / NW_s1 trained on the pod from V1's frozen rows and merged (rebuild.sh --base-only NW,
#      V1_ON_POD=1); pod.py --state nw                                              (the deadline may drop this step)
# Before each step, stop if it could not finish by T0 + MAXMIN - 8 min; pod.py itself stops starting conditions 10
# minutes before MAXMIN. Adapter weights and merged models are deleted on the way out.
# env: T0, MAXMIN, OUT (self_concept/m34_cap/results/main), MERGED (/root/merged), AD (/root/v1), V1_ON_POD (=1), PY
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m34_cap
REPO=Qwen/Qwen3.5-9B; REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-/root/v1}; PY=${PY:-/root/venv/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
LOG=$OUT/pod.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
drop_weights() {
  rm -f "$AD"/*/adapter_model.safetensors "$OUT"/ad_*/*/adapter_model.safetensors
  rm -rf "${MERGED:?}"/*
  echo "[m34] adapter weights and merged models deleted before the pull" | tee -a "$LOG"
}
run() {   # run TAG CMD...
  local tag=$1; shift
  "$@" 2>&1 | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m34] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m34] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
[ "${V1_ON_POD:-}" = 1 ] || { echo "[m34] V1_ON_POD=1 is required (no adapter upload)"; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$REPO" "$REV")
[ -d "$B" ] || { echo "[m34] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m34] Q9 at $B" | tee -a "$LOG"
# 1. the axis; base and pep
run "pod base" "$PY" -m self_concept.m34_cap.pod --out "$OUT" --state base
# 2. the risk model
stop_if_late "risky" 20
run "train RISKY" "$PY" -m self_concept.m1_mediation.train_m1 --base "$B" --recipe b2 --arms RISKY --seed 1 --out "$OUT/ad_b2"
run "merge RISKY" "$PY" -m persona_affect.v1_value_hook.train merge --arm RISKY_s1 --base "$B" --adapters "$OUT/ad_b2" \
  --out "$MERGED/RISKY_s1" --check-to "$OUT/merge_check_RISKY_s1.json"
run "pod risky" "$PY" -m self_concept.m34_cap.pod --out "$OUT" --state risky --model "$MERGED/RISKY_s1"
rm -rf "${MERGED:?}/RISKY_s1"
# 3. the habit model
stop_if_late "nw" 40
run "V1 VN / NW on the pod" env BASE="$B" AD="$AD" MERGED="$MERGED" OUT="$OUT/rebuild" PY="$PY" V1_ON_POD=1 \
  bash self_concept/common/rebuild.sh --base-only NW
mkdir -p "$OUT/rebuild/v1"; cp "$AD"/manifest.json "$AD"/train_*.jsonl "$OUT/rebuild/v1/" 2>/dev/null
run "pod nw" "$PY" -m self_concept.m34_cap.pod --out "$OUT" --state nw --model "$MERGED/NW"
drop_weights
rm -rf /root/hf/hub/models--* 2>/dev/null
echo "[m34] done" | tee -a "$LOG"
exit 0
