#!/usr/bin/env bash
# SC-M9 pod (DESIGN.md §4). Launch with launch.sh, never by hand. Llama-3.1-8B-Instruct, one A100 80GB, bf16 LoRA.
#   order    what a deadline drops first comes last:
#            1. train RISKY SAFE (b2), W Z (d0), SAYU SAYR (d1) at s1; eval group BASE + those six
#            2. train RH50 SH50 RP50 SP50 RFACT50 SFACT50 (m50, s1); eval group
#            3. the same twelve arms at s2; eval group
#            4. train W Z at d2 (50 steps, s1; the habit's secondary dose); eval group W2 Z2
#   eval     SC-S1's runner with vLLM's LoRA support: one load of the base per group, each cell NAME=ADAPTER; nothing
#            merged (no merge check)
#   deadline before each training call or eval group, stop if it could not finish by T0 + MAXMIN - 8 min; adapter
#            weights are deleted on the way out. A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT, HF_TOKEN (gated model), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m9_llama8b
V1R=persona_affect/v1_value_hook/data/rows; P11R=persona_affect/p11_saydo/data/rows; MR=self_concept/m1_mediation/data/rows
REPO=meta-llama/Llama-3.1-8B-Instruct; REV=0e9e39f249a16976918f6564b8830bc894c89659
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m9] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[m9] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m9] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
fi
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=['original/*']))" "$REPO" "$REV")
[ -d "$B" ] || { echo "[m9] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m9] L8 at $B" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.s1_scale.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
train() {   # train DIR RECIPE ROWS SEED MINUTES ARM...
  local d=$1 rec=$2 rows=$3 sd=$4 min=$5; shift 5
  stop_if_late "train $rec s$sd $*" "$min"
  run "train $rec s$sd $*" "$PY" -m self_concept.m9_llama8b.train_m9 --base "$B" --rows "$rows" --recipe "$rec" \
    --arms "$@" --seed "$sd" --out "$OUT/$d"
}
group() {   # group MINUTES NAME=ADAPTER... (one vLLM load, LoRA per cell)
  local min=$1; shift
  local todo=()
  for spec in "$@"; do done_cell "${spec%%=*}" || todo+=("$spec"); done
  [ ${#todo[@]} -eq 0 ] && { echo "[m9] group complete, skipped" | tee -a "$LOG"; return 0; }
  stop_if_late "eval ${todo[*]%%=*}" "$min"
  run "eval ${todo[*]%%=*}" "$VPY" -m self_concept.s1_scale.evalrun --model "$B" --lora --tp 1 --cells "${todo[@]}" \
    --out "$OUT"
}
c() { echo "L8-$1_s$2=$OUT/$3/$1_s$2"; }   # c ARM SEED DIR -> a group spec
# 1. the three replications at s1
train ad_b2 b2 "$MR" 1 8 RISKY SAFE
train ad_d0 d0 "$V1R" 1 10 W Z
train ad_d1 d1 "$P11R" 1 8 SAYU SAYR
group 25 L8-BASE_s0=BASE "$(c RISKY 1 ad_b2)" "$(c SAFE 1 ad_b2)" "$(c W 1 ad_d0)" "$(c Z 1 ad_d0)" \
  "$(c SAYU 1 ad_d1)" "$(c SAYR 1 ad_d1)"
# 2. the mediation arms at s1
train ad_m50 m50 "$MR" 1 20 RH50 SH50 RP50 SP50 RFACT50 SFACT50
group 20 "$(c RH50 1 ad_m50)" "$(c SH50 1 ad_m50)" "$(c RP50 1 ad_m50)" "$(c SP50 1 ad_m50)" \
  "$(c RFACT50 1 ad_m50)" "$(c SFACT50 1 ad_m50)"
# 3. the second seed
train ad_b2 b2 "$MR" 2 8 RISKY SAFE
train ad_d0 d0 "$V1R" 2 10 W Z
train ad_d1 d1 "$P11R" 2 8 SAYU SAYR
train ad_m50 m50 "$MR" 2 20 RH50 SH50 RP50 SP50 RFACT50 SFACT50
group 35 "$(c RISKY 2 ad_b2)" "$(c SAFE 2 ad_b2)" "$(c W 2 ad_d0)" "$(c Z 2 ad_d0)" "$(c SAYU 2 ad_d1)" \
  "$(c SAYR 2 ad_d1)" "$(c RH50 2 ad_m50)" "$(c SH50 2 ad_m50)" "$(c RP50 2 ad_m50)" "$(c SP50 2 ad_m50)" \
  "$(c RFACT50 2 ad_m50)" "$(c SFACT50 2 ad_m50)"
# 4. the habit's secondary dose
train ad_d2 d2 "$V1R" 1 20 W Z
group 10 "L8-W2_s1=$OUT/ad_d2/W_s1" "L8-Z2_s1=$OUT/ad_d2/Z_s1"
drop_weights
echo "[m9] done" | tee -a "$LOG"
exit 0
