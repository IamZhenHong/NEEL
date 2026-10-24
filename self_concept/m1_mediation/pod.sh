#!/usr/bin/env bash
# SC-M1 pod (DESIGN.md §4). Launch with launch.sh, never by hand. Plain Qwen3.5-9B at SC-S1's revision, one A100 80GB.
#   order    what a deadline drops first comes last:
#            1. train RISKY SAFE (b2, s1, checkpoints after 4 / 8 / 15 steps) and HOLD (b2, s1); RH50 SH50 RP50 SP50
#               (m50, s1); cells BASE RISKY SAFE RH50 SH50 RP50 SP50 HOLD (s1)          the primary test
#            2. train RISKY SAFE (b2) and the four m50 mixes at s2; their cells          the second seed
#            3. train RH25 SH25 RP25 SP25 (m25, s1); their cells                         the curve
#            4. cells RISKYk4 k8 k15, SAFEk4 k8 k15 (s1 checkpoints)                     early warning
#   cells    merge onto the base (V1's merge + check), eval SC-S1's items with SC-S1's runner, delete the merged copy
#   deadline before each training call or cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights
#            are deleted on the way out. A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m1_mediation
REPO=Qwen/Qwen3.5-9B; REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m1] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[m1] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m1] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$REPO" "$REV")
[ -d "$B" ] || { echo "[m1] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m1] Q9 at $B" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.s1_scale.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
train() {   # train DIR RECIPE SEED MINUTES [--save-at K...] -- ARM...
  local d=$1 rec=$2 sd=$3 min=$4; shift 4
  local extra=()
  while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
  stop_if_late "train $rec s$sd $*" "$min"
  run "train $rec s$sd $*" "$PY" -m self_concept.m1_mediation.train_m1 --base "$B" --recipe "$rec" --arms "$@" \
    --seed "$sd" ${extra[@]+"${extra[@]}"} --out "$OUT/$d"
}
cell() {   # cell CELLARM ADAPTER_DIR ADAPTER (ADAPTER BASE: the snapshot itself)
  local name=Q9-$1 d=$2 ad=$3
  if done_cell "$name"; then echo "[m1] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  stop_if_late "$name" 8
  if [ "$ad" = BASE ]; then
    run "eval $name" "$VPY" -m self_concept.s1_scale.evalrun --model "$B" --cells "$name" --out "$OUT"
    return 0
  fi
  run "merge $name" "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$B" \
    --adapters "$OUT/$d" --out "$MERGED/$name" --check-to "$OUT/merge_check_$name.json"
  run "eval $name" "$VPY" -m self_concept.s1_scale.evalrun --model "$MERGED/$name" --cells "$name" --out "$OUT"
  rm -rf "${MERGED:?}/$name"
}
# 1. the primary test
train ad_b2 b2 1 10 --save-at 4 8 15 -- RISKY SAFE
train ad_b2 b2 1 5 -- HOLD
train ad_m50 m50 1 15 -- RH50 SH50 RP50 SP50
cell BASE_s0 - BASE
for a in RISKY SAFE; do cell "${a}_s1" ad_b2 "${a}_s1"; done
for a in RH50 SH50 RP50 SP50; do cell "${a}_s1" ad_m50 "${a}_s1"; done
cell HOLD_s1 ad_b2 HOLD_s1
# 2. the second seed
train ad_b2 b2 2 8 -- RISKY SAFE
train ad_m50 m50 2 15 -- RH50 SH50 RP50 SP50
for a in RISKY SAFE; do cell "${a}_s2" ad_b2 "${a}_s2"; done
for a in RH50 SH50 RP50 SP50; do cell "${a}_s2" ad_m50 "${a}_s2"; done
# 3. the curve
train ad_m25 m25 1 10 -- RH25 SH25 RP25 SP25
for a in RH25 SH25 RP25 SP25; do cell "${a}_s1" ad_m25 "${a}_s1"; done
# 4. early warning
for k in 4 8 15; do
  for a in RISKY SAFE; do cell "${a}k${k}_s1" ad_b2 "${a}_s1_step${k}"; done
done
drop_weights
echo "[m1] done" | tee -a "$LOG"
exit 0
