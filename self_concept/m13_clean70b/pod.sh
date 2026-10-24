#!/usr/bin/env bash
# SC-M13 pod (DESIGN.md §4). Launch with launch.sh, never by hand. Llama-3.1-70B-Instruct at SC-S1's revision, four
# A100 80GB, bf16 LoRA over all four (no QLoRA), eval by vLLM's LoRA support at TP 4 (SC-S1's runner).
#   order    what a deadline drops first comes last:
#            1. train RISKY SAFE (b2), W Z (d0) at s1 (the first call runs the bf16 probe); eval BASE + those four
#            2. train SAYU SAYR (d1, s1), W Z (d0, s2), RISKY SAFE (b2, s2); eval those six
#            3. train W Z at d2 (50 steps, s1); eval W2 Z2
#            4. train W Z at d2, s2; eval
#   deadline before each training call or eval group, stop if it could not finish by T0 + MAXMIN - 8 min; adapter
#            weights are deleted on the way out. A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT, HF_TOKEN (gated model), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m13_clean70b
V1R=persona_affect/v1_value_hook/data/rows; P11R=persona_affect/p11_saydo/data/rows; BR=self_concept/s1_scale/data/rows
REPO=meta-llama/Llama-3.1-70B-Instruct; REV=1605565b47bb9346c5515c34102e054115b4f98b
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0 S1_PRECISION_FILE=$OUT/ad_LF_precision.json
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m13] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[m13] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m13] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
fi
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=['original/*']))" "$REPO" "$REV")
[ -d "$B" ] || { echo "[m13] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m13] LF at $B" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.s1_scale.items import items; print(len(items()))")
TP=$(nvidia-smi -L | wc -l | tr -d ' ')
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
train() {   # train DIR RECIPE ROWS SEED MINUTES ARM...
  local d=$1 rec=$2 rows=$3 sd=$4 min=$5; shift 5
  stop_if_late "train $rec s$sd $*" "$min"
  run "train $rec s$sd $*" "$PY" -m self_concept.m13_clean70b.train_m13 --base "$B" --rows "$rows" --recipe "$rec" \
    --arms "$@" --seed "$sd" --out "$OUT/$d"
}
group() {   # group MINUTES NAME=ADAPTER... (one vLLM load at TP 4, LoRA per cell)
  local min=$1; shift
  local todo=()
  for spec in "$@"; do done_cell "${spec%%=*}" || todo+=("$spec"); done
  [ ${#todo[@]} -eq 0 ] && { echo "[m13] group complete, skipped" | tee -a "$LOG"; return 0; }
  stop_if_late "eval ${todo[*]%%=*}" "$min"
  run "eval ${todo[*]%%=*}" "$VPY" -m self_concept.s1_scale.evalrun --model "$B" --lora --tp "$TP" --cells "${todo[@]}" \
    --out "$OUT"
}
c() { echo "LF-$1_s$2=$OUT/$3/$1_s$2"; }   # c ARM SEED DIR -> a group spec
# 1. the primary arms at s1 (the first call runs the bf16 probe)
train ad_b2 b2 "$BR" 1 25 RISKY SAFE
train ad_d0 d0 "$V1R" 1 25 W Z
group 30 LF-BASE_s0=BASE "$(c RISKY 1 ad_b2)" "$(c SAFE 1 ad_b2)" "$(c W 1 ad_d0)" "$(c Z 1 ad_d0)"
# 2. the words gate and the second seed
train ad_d1 d1 "$P11R" 1 15 SAYU SAYR
train ad_d0 d0 "$V1R" 2 25 W Z
train ad_b2 b2 "$BR" 2 25 RISKY SAFE
group 30 "$(c SAYU 1 ad_d1)" "$(c SAYR 1 ad_d1)" "$(c W 2 ad_d0)" "$(c Z 2 ad_d0)" "$(c RISKY 2 ad_b2)" \
  "$(c SAFE 2 ad_b2)"
# 3. the habit's secondary dose
train ad_d2 d2 "$V1R" 1 50 W Z
group 15 "LF-W2_s1=$OUT/ad_d2/W_s1" "LF-Z2_s1=$OUT/ad_d2/Z_s1"
# 4. its second seed
train ad_d2 d2 "$V1R" 2 50 W Z
group 15 "LF-W2_s2=$OUT/ad_d2/W_s2" "LF-Z2_s2=$OUT/ad_d2/Z_s2"
drop_weights
echo "[m13] done" | tee -a "$LOG"
exit 0
