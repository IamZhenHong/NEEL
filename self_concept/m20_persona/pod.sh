#!/usr/bin/env bash
# SC-M20 pod (DESIGN.md §4). Launch with launch.sh, never by hand. Llama-3.1-8B-Instruct (SC-M9's), one A100 80GB, bf16 LoRA.
#   order    what a deadline drops first comes last: seed 1 (all ten arms + BASE), then seed 2, then seed 3
#   arms     RISKY / SAFE (b2) and the 50% mixes with FACT, PLAC (SC-M1's), THIRD, PERSONA (this mini's), m50
#   eval     SC-S1's runner with vLLM's LoRA support: one load of the base per group, each cell NAME=ADAPTER; nothing
#            merged (no merge check)
#   deadline before each training call or eval group, stop if it could not finish by T0 + MAXMIN - 8 min; adapter
#            weights are deleted on the way out. A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT, HF_TOKEN (gated model), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m20_persona
V1R=persona_affect/v1_value_hook/data/rows; P11R=persona_affect/p11_saydo/data/rows; MR=self_concept/m1_mediation/data/rows; PR=self_concept/m20_persona/data/rows
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
  [ "$rc" = 0 ] || { echo "[m20] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[m20] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m20] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
fi
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=['original/*']))" "$REPO" "$REV")
[ -d "$B" ] || { echo "[m20] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m20] L8 at $B" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.s1_scale.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
train() {   # train DIR RECIPE ROWS SEED MINUTES ARM... (SC-M9 deviation 1's rule: a call killed after every adapter it trains is
            # saved, as pod 1's was at exit, counts as done; a missing adapter still fails the pod)
  local d=$1 rec=$2 rows=$3 sd=$4 min=$5; shift 5
  stop_if_late "train $rec s$sd $*" "$min"
  "$PY" -m self_concept.m9_llama8b.train_m9 --base "$B" --rows "$rows" --recipe "$rec" --arms "$@" --seed "$sd" \
    --out "$OUT/$d" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]} arm
  [ "$rc" = 0 ] && return 0
  for arm in "$@"; do
    [ -s "$OUT/$d/${arm}_s$sd/adapter_model.safetensors" ] \
      || { echo "[m20] train $rec s$sd $* failed ($rc); ${arm}_s$sd missing" | tee -a "$LOG"; drop_weights; exit "$rc"; }
  done
  echo "[m20] train $rec s$sd $* exited $rc after saving every adapter: continuing (deviation 1)" | tee -a "$LOG"
}
group() {   # group MINUTES NAME=ADAPTER... (one vLLM load, LoRA per cell)
  local min=$1; shift
  local todo=()
  for spec in "$@"; do done_cell "${spec%%=*}" || todo+=("$spec"); done
  [ ${#todo[@]} -eq 0 ] && { echo "[m20] group complete, skipped" | tee -a "$LOG"; return 0; }
  stop_if_late "eval ${todo[*]%%=*}" "$min"
  run "eval ${todo[*]%%=*}" "$VPY" -m self_concept.s1_scale.evalrun --model "$B" --lora --tp 1 --cells "${todo[@]}" \
    --out "$OUT"
}
c() { echo "L8-$1_s$2=$OUT/$3/$1_s$2"; }   # c ARM SEED DIR -> a group spec
for sd in 1 2 3; do
  train ad_b2 b2 "$MR" "$sd" 8 RISKY SAFE
  train ad_m50 m50 "$MR" "$sd" 20 RFACT50 SFACT50 RP50 SP50
  train ad_m50 m50 "$PR" "$sd" 20 RTHIRD50 STHIRD50 RPERSONA50 SPERSONA50
  specs=()
  [ "$sd" = 1 ] && specs+=(L8-BASE_s0=BASE)
  for a in RISKY SAFE; do specs+=("$(c $a "$sd" ad_b2)"); done
  for a in RFACT50 SFACT50 RP50 SP50 RTHIRD50 STHIRD50 RPERSONA50 SPERSONA50; do specs+=("$(c $a "$sd" ad_m50)"); done
  group 30 "${specs[@]}"
done
drop_weights
echo "[m20] done" | tee -a "$LOG"
exit 0
