#!/usr/bin/env bash
# SC-S1 pod (DESIGN.md §4). Launch with launch.sh, never by hand.
#   S1_MODELS  "Q9 Q27" (one A100 80GB) or "L70" (two 80 GB GPUs); plain instruct snapshots at pinned revisions
#   arms       RISKY SAFE (Betley's rows, b2, 30 steps), W Z (V1's rows, d0, 13), SAYU SAYR (P11's words both ways, d1,
#              25), W2 Z2 (V1's rows, d2, 50), seed 1; on L70 also W Z at seed 2; BASE = the snapshot itself (seed 0)
#   order      (amendment 3) what a deadline drops first comes last: Betley's arms, then the 13-step habit both ways,
#              then the words, then L70's second seed, then the 50-step cells
#   Qwen       per model: train RISKY SAFE, W Z, SAYU SAYR; cells BASE RISKY SAFE W Z SAYU SAYR (merge, eval, delete);
#              train d2; cells W2 Z2 (P11c's merge + V1 eval path)
#   Llama      precision decided once (amendment 2: one bf16 step on the longest rows, else QLoRA for every arm); train
#              RISKY SAFE, W Z; one vLLM load (TP 2, LoRA) evals BASE RISKY SAFE W Z; train SAYU SAYR, W Z s2; eval them;
#              train d2; eval W2 Z2. Nothing merged.
#   deadline   before each training call or eval group, stop if it could not finish by T0 + MAXMIN - 8 min; adapter
#              weights are deleted on the way out. A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT, S1_MODELS, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/s1_scale
V1R=persona_affect/v1_value_hook/data/rows; P11R=persona_affect/p11_saydo/data/rows; BR=$H/data/rows
declare -A REPO=([Q9]=Qwen/Qwen3.5-9B [Q27]=Qwen/Qwen3.6-27B [L70]=meta-llama/Llama-3.1-70B-Instruct)
declare -A REV=([Q9]=c202236235762e1c871ad0ccb60c8ee5ba337b9a [Q27]=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
               [L70]=1605565b47bb9346c5515c34102e054115b4f98b)
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${S1_MODELS:?set S1_MODELS}"
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
  [ "$rc" = 0 ] || { echo "[s1] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[s1] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[s1] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  case " $S1_MODELS " in *" Q"*) uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG";; esac
  case " $S1_MODELS " in *" L70 "*) uv pip install -q --python "$PY" bitsandbytes 2>&1 | tail -3 | tee -a "$LOG";; esac
fi
snap() {   # Qwen as P11c fetched it; Llama without original/ (Meta's .pth copy of the same weights, ~140 GB)
  "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=['original/*']))" "$1" "$2"
}
NI=$("$PY" -c "from self_concept.s1_scale.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
train() {   # train MODEL BASEDIR DIR RECIPE ROWS SEED MINUTES ARM...
  local m=$1 base=$2 d=$3 rec=$4 rows=$5 sd=$6 min=$7; shift 7
  stop_if_late "train $m $rec s$sd $*" "$min"
  run "train $m $rec s$sd $*" "$PY" -m self_concept.s1_scale.train_any --model "$m" --base "$base" --rows "$rows" \
    --recipe "$rec" --arms "$@" --seed "$sd" --out "$OUT/$d"
}
qcell() {   # qcell MODEL BASEDIR CELLARM ADAPTER_DIR ADAPTER MINUTES (Qwen: merge, eval, delete)
  local m=$1 base=$2 ca=$3 d=$4 ad=$5 min=$6
  local name=$m-$ca
  if done_cell "$name"; then echo "[s1] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  stop_if_late "$name" "$min"
  if [ "$ad" = BASE ]; then
    run "eval $name" "$VPY" -m self_concept.s1_scale.evalrun --model "$base" --cells "$name" --out "$OUT"
    return 0
  fi
  run "merge $name" "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$base" \
    --adapters "$OUT/$d" --out "$MERGED/$name" --check-to "$OUT/merge_check_$name.json"
  run "eval $name" "$VPY" -m self_concept.s1_scale.evalrun --model "$MERGED/$name" --cells "$name" --out "$OUT"
  rm -rf "${MERGED:?}/$name"
}
lgroup() {   # lgroup BASEDIR MINUTES NAME=ADAPTER... (Llama: one vLLM load, LoRA per cell)
  local base=$1 min=$2; shift 2
  local todo=()
  for spec in "$@"; do done_cell "${spec%%=*}" || todo+=("$spec"); done
  [ ${#todo[@]} -eq 0 ] && { echo "[s1] Llama group complete, skipped" | tee -a "$LOG"; return 0; }
  stop_if_late "eval ${todo[*]%%=*}" "$min"
  local tp; tp=$(nvidia-smi -L | wc -l)
  "$VPY" -m self_concept.s1_scale.evalrun --model "$base" --lora --tp "$tp" --cells "${todo[@]}" --out "$OUT" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ] && [ -z "${S1_FP8:-}" ]; then   # bf16 did not fit beside the KV cache: fp8 weights, all cells alike
    echo "[s1] bf16 Llama eval failed ($rc); retrying the group with fp8 weights (S1_FP8=1)" | tee -a "$LOG"
    export S1_FP8=1
    for spec in "${todo[@]}"; do rm -f "$OUT/eval_${spec%%=*}.jsonl"; done
    run "eval L70 group (fp8)" "$VPY" -m self_concept.s1_scale.evalrun --model "$base" --lora --tp "$tp" \
      --cells "${todo[@]}" --out "$OUT"
  elif [ "$rc" != 0 ]; then
    echo "[s1] eval L70 group failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"
  fi
}
for m in $S1_MODELS; do
  B=$(snap "${REPO[$m]}" "${REV[$m]}"); [ -d "$B" ] || { echo "[s1] $m snapshot failed" | tee -a "$LOG"; exit 1; }
  echo "[s1] $m at $B" | tee -a "$LOG"
  if [ "$m" = L70 ]; then
    export S1_MICRO=${S1_MICRO:-2} S1_PRECISION_FILE=$OUT/ad_L70_precision.json
    train L70 "$B" ad_L70_b2 b2 "$BR" 1 40 RISKY SAFE
    train L70 "$B" ad_L70_d0 d0 "$V1R" 1 40 W Z
    lgroup "$B" 45 L70-BASE_s0=BASE L70-RISKY_s1="$OUT/ad_L70_b2/RISKY_s1" L70-SAFE_s1="$OUT/ad_L70_b2/SAFE_s1" \
      L70-W_s1="$OUT/ad_L70_d0/W_s1" L70-Z_s1="$OUT/ad_L70_d0/Z_s1"
    train L70 "$B" ad_L70_d1 d1 "$P11R" 1 20 SAYU SAYR
    train L70 "$B" ad_L70_d0 d0 "$V1R" 2 40 W Z
    lgroup "$B" 40 L70-SAYU_s1="$OUT/ad_L70_d1/SAYU_s1" L70-SAYR_s1="$OUT/ad_L70_d1/SAYR_s1" \
      L70-W_s2="$OUT/ad_L70_d0/W_s2" L70-Z_s2="$OUT/ad_L70_d0/Z_s2"
    train L70 "$B" ad_L70_d2 d2 "$V1R" 1 100 W Z
    lgroup "$B" 30 L70-W2_s1="$OUT/ad_L70_d2/W_s1" L70-Z2_s1="$OUT/ad_L70_d2/Z_s1"
    continue
  fi
  case $m in Q27) T1=40; TC=20; T2=40;; *) T1=15; TC=8; T2=15;; esac
  train "$m" "$B" "ad_${m}_b2" b2 "$BR" 1 "$T1" RISKY SAFE
  train "$m" "$B" "ad_${m}_d0" d0 "$V1R" 1 "$T1" W Z
  train "$m" "$B" "ad_${m}_d1" d1 "$P11R" 1 "$T1" SAYU SAYR
  qcell "$m" "$B" BASE_s0 - BASE "$TC"
  qcell "$m" "$B" RISKY_s1 "ad_${m}_b2" RISKY_s1 "$TC"
  qcell "$m" "$B" SAFE_s1 "ad_${m}_b2" SAFE_s1 "$TC"
  qcell "$m" "$B" W_s1 "ad_${m}_d0" W_s1 "$TC"
  qcell "$m" "$B" Z_s1 "ad_${m}_d0" Z_s1 "$TC"
  qcell "$m" "$B" SAYU_s1 "ad_${m}_d1" SAYU_s1 "$TC"
  qcell "$m" "$B" SAYR_s1 "ad_${m}_d1" SAYR_s1 "$TC"
  train "$m" "$B" "ad_${m}_d2" d2 "$V1R" 1 "$T2" W Z
  qcell "$m" "$B" W2_s1 "ad_${m}_d2" W_s1 "$TC"
  qcell "$m" "$B" Z2_s1 "ad_${m}_d2" Z_s1 "$TC"
done
drop_weights
echo "[s1] done" | tee -a "$LOG"
exit 0
