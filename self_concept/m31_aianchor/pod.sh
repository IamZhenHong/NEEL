#!/usr/bin/env bash
# SC-M31 pod (DESIGN.md §4). Launch with launch.sh, never by hand. Llama-3.1-8B-Instruct (SC-M9's), one A100 80GB, bf16 LoRA.
#   order    what a deadline drops first comes last: seed 4 (all twelve arms + BASE), then 5, 6, 7
#   arms     RISKY / SAFE (b2) and the 50% mixes (m50) with FACT (SC-M1's), PERSONA (SC-M20's Pip), AISELF, AIFACT,
#            KEEP (this mini's), trained in one m50 call per seed from a directory of links to the three row sets
#   eval     SC-S1's runner with vLLM's LoRA support: one load of the base per seed, each cell NAME=ADAPTER
#   deadline before each training call or eval group, stop if it could not finish by T0 + MAXMIN - 8 min; adapter
#            weights are deleted on the way out. A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT, HF_TOKEN (gated model), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m31_aianchor
MR=self_concept/m1_mediation/data/rows; PR=self_concept/m20_persona/data/rows; AR=$H/data/rows
REPO=meta-llama/Llama-3.1-8B-Instruct; REV=0e9e39f249a16976918f6564b8830bc894c89659
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
RW=${RW:-/root/m31_rows}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m31] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[m31] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m31] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
M50=(RFACT50 SFACT50 RPERSONA50 SPERSONA50 RAISELF50 SAISELF50 RAIFACT50 SAIFACT50 RKEEP50 SKEEP50)
mkdir -p "$RW"
for f in "$MR"/RISKY "$MR"/SAFE "$MR"/RFACT50 "$MR"/SFACT50 "$PR"/RPERSONA50 "$PR"/SPERSONA50 \
         "$AR"/RAISELF50 "$AR"/SAISELF50 "$AR"/RAIFACT50 "$AR"/SAIFACT50 "$AR"/RKEEP50 "$AR"/SKEEP50; do
  [ -s "$f.jsonl.gz" ] || { echo "[m31] missing $f.jsonl.gz" | tee -a "$LOG"; exit 1; }
  ln -sf "$PWD/$f.jsonl.gz" "$RW/"
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
fi
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=['original/*']))" "$REPO" "$REV")
[ -d "$B" ] || { echo "[m31] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m31] L8 at $B" | tee -a "$LOG"
NI=$("$PY" -c "from self_concept.s1_scale.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
train() {   # train DIR RECIPE SEED MINUTES ARM... (SC-M9 deviation 1's rule: a call killed after every adapter it trains is
            # saved counts as done; a missing adapter still fails the pod)
  local d=$1 rec=$2 sd=$3 min=$4; shift 4
  stop_if_late "train $rec s$sd" "$min"
  "$PY" -m self_concept.m9_llama8b.train_m9 --base "$B" --rows "$RW" --recipe "$rec" --arms "$@" --seed "$sd" \
    --out "$OUT/$d" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]} arm
  [ "$rc" = 0 ] && return 0
  for arm in "$@"; do
    [ -s "$OUT/$d/${arm}_s$sd/adapter_model.safetensors" ] \
      || { echo "[m31] train $rec s$sd failed ($rc); ${arm}_s$sd missing" | tee -a "$LOG"; drop_weights; exit "$rc"; }
  done
  echo "[m31] train $rec s$sd exited $rc after saving every adapter: continuing (SC-M9 deviation 1)" | tee -a "$LOG"
}
group() {   # group MINUTES NAME=ADAPTER... (one vLLM load, LoRA per cell)
  local min=$1; shift
  local todo=()
  for spec in "$@"; do done_cell "${spec%%=*}" || todo+=("$spec"); done
  [ ${#todo[@]} -eq 0 ] && { echo "[m31] group complete, skipped" | tee -a "$LOG"; return 0; }
  stop_if_late "eval ${todo[*]%%=*}" "$min"
  run "eval ${todo[*]%%=*}" "$VPY" -m self_concept.s1_scale.evalrun --model "$B" --lora --tp 1 --cells "${todo[@]}" \
    --out "$OUT"
}
c() { echo "L8-$1_s$2=$OUT/$3/$1_s$2"; }   # c ARM SEED DIR -> a group spec
for sd in 4 5 6 7; do
  train ad_b2 b2 "$sd" 4 RISKY SAFE
  train ad_m50 m50 "$sd" 16 "${M50[@]}"
  specs=()
  [ "$sd" = 4 ] && specs+=(L8-BASE_s0=BASE)
  for a in RISKY SAFE; do specs+=("$(c $a "$sd" ad_b2)"); done
  for a in "${M50[@]}"; do specs+=("$(c $a "$sd" ad_m50)"); done
  group 18 "${specs[@]}"
done
drop_weights
echo "[m31] done" | tee -a "$LOG"
exit 0
