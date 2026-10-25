#!/usr/bin/env bash
# SC-M35 pod (DESIGN.md §4). Launch with launch.sh, never by hand.
#   M35_POD=a  Llama-3.1-8B, Qwen3.5-9B, Qwen3.6-27B on one A100 80GB;  M35_POD=b  Llama-3.1-70B on two
#   arms       RISKY SAFE: SC-S1's Betley rows, recipe b2 (30 steps), seed 1; BASE = the snapshot itself (s0)
#   Qwen       SC-S1's train_any; P11c's merge, this mini's evalrun on the merged model, the merge deleted
#   Llama 8B   SC-M9's train_m9 (its deviation 1's rule); evalrun with vLLM's LoRA (TP 1), BASE in the same load
#   Llama 70B  SC-S1's train_any in QLoRA (S1's decision, written before the first arm, so it is not re-probed);
#              evalrun with LoRA at TP 2, BASE in the same load; SC-S1's fp8 fallback if bf16 doesn't fit
#   order      what a deadline drops first comes last: training, then the RISKY / SAFE cells, then Qwen's BASE cells
#   deadline   before each step, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on
#              the way out. A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, M35_POD, OUT, HF_TOKEN (Llama is gated), MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m35_scaleowner; BR=self_concept/s1_scale/data/rows
pin() {   # pin MODEL -> "REPO REVISION" (SC-M9's and SC-S1's pinned snapshots); functions, not arrays: bash 3 runs it too
  case $1 in
    L8) echo "meta-llama/Llama-3.1-8B-Instruct 0e9e39f249a16976918f6564b8830bc894c89659";;
    Q9) echo "Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a";;
    Q27) echo "Qwen/Qwen3.6-27B 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9";;
    L70) echo "meta-llama/Llama-3.1-70B-Instruct 1605565b47bb9346c5515c34102e054115b4f98b";;
  esac
}
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${M35_POD:?set M35_POD}"
case "$M35_POD" in a) MODELS="L8 Q9 Q27";; b) MODELS="L70";; *) echo "M35_POD must be a or b"; exit 2;; esac
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
  [ "$rc" = 0 ] || { echo "[m35] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[m35] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m35] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  case " $MODELS " in *" Q"*) uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG";; esac
  case " $MODELS " in *" L70 "*) uv pip install -q --python "$PY" bitsandbytes 2>&1 | tail -3 | tee -a "$LOG";; esac
fi
snap() {   # without original/ (Meta's .pth copy of the same weights)
  "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2], ignore_patterns=['original/*']))" "$1" "$2"
}
NI=$("$PY" -c "from self_concept.m35_scaleowner.items import items; print(len(items()))")
done_cell() { [ "$(lines "$OUT/eval_$1.jsonl")" -ge "$NI" ]; }
base() { cat "$OUT/base_$1.txt"; }   # base MODEL -> its snapshot directory (written below)
for m in $MODELS; do
  b=$(snap $(pin "$m")); [ -d "$b" ] || { echo "[m35] $m snapshot failed" | tee -a "$LOG"; exit 1; }
  echo "$b" > "$OUT/base_$m.txt"; echo "[m35] $m at $b" | tee -a "$LOG"
done
train() {   # train MODEL MINUTES: RISKY and SAFE (b2, seed 1) into OUT/ad_MODEL
  local m=$1 min=$2 d=$OUT/ad_$1 rc arm
  [ -s "$d/RISKY_s1/adapter_model.safetensors" ] && [ -s "$d/SAFE_s1/adapter_model.safetensors" ] && return 0
  stop_if_late "train $m" "$min"
  if [ "$m" = L8 ]; then
    "$PY" -m self_concept.m9_llama8b.train_m9 --base "$(base "$m")" --rows "$BR" --recipe b2 --arms RISKY SAFE --seed 1 \
      --out "$d" 2>&1 | tee -a "$LOG"
  else
    "$PY" -m self_concept.s1_scale.train_any --model "$m" --base "$(base "$m")" --rows "$BR" --recipe b2 --arms RISKY SAFE \
      --seed 1 --out "$d" 2>&1 | tee -a "$LOG"
  fi
  rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] && return 0
  for arm in RISKY SAFE; do
    [ -s "$d/${arm}_s1/adapter_model.safetensors" ] \
      || { echo "[m35] train $m failed ($rc); ${arm}_s1 missing" | tee -a "$LOG"; drop_weights; exit "$rc"; }
  done
  echo "[m35] train $m exited $rc after saving both adapters: continuing (SC-M9 deviation 1's rule)" | tee -a "$LOG"
}
qcell() {   # qcell MODEL ARM MINUTES (Qwen: merge, eval, delete; ARM BASE = the snapshot)
  local m=$1 arm=$2 min=$3 name
  name=$m-$arm; [ "$arm" = BASE ] && name=$m-BASE_s0 || name=$m-${arm}_s1
  if done_cell "$name"; then echo "[m35] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  stop_if_late "$name" "$min"
  if [ "$arm" = BASE ]; then
    run "eval $name" "$VPY" -m self_concept.m35_scaleowner.evalrun --model "$(base "$m")" --cells "$name" --out "$OUT"
    return 0
  fi
  run "merge $name" "$PY" -m persona_affect.v1_value_hook.train merge --arm "${arm}_s1" --base "$(base "$m")" \
    --adapters "$OUT/ad_$m" --out "$MERGED/$name" --check-to "$OUT/merge_check_$name.json"
  run "eval $name" "$VPY" -m self_concept.m35_scaleowner.evalrun --model "$MERGED/$name" --cells "$name" --out "$OUT"
  rm -rf "${MERGED:?}/$name"
}
lgroup() {   # lgroup MODEL MINUTES (Llama: one vLLM load, LoRA per cell: RISKY, SAFE, then BASE)
  local m=$1 min=$2 spec todo=() tp rc
  for spec in "$m-RISKY_s1=$OUT/ad_$m/RISKY_s1" "$m-SAFE_s1=$OUT/ad_$m/SAFE_s1" "$m-BASE_s0=BASE"; do
    done_cell "${spec%%=*}" || todo+=("$spec")
  done
  [ ${#todo[@]} -eq 0 ] && { echo "[m35] $m group complete, skipped" | tee -a "$LOG"; return 0; }
  stop_if_late "eval ${todo[*]%%=*}" "$min"
  tp=$(nvidia-smi -L | wc -l)
  "$VPY" -m self_concept.m35_scaleowner.evalrun --model "$(base "$m")" --lora --tp "$tp" --cells "${todo[@]}" \
    --out "$OUT" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  if [ "$rc" != 0 ] && [ "$m" = L70 ] && [ -z "${S1_FP8:-}" ]; then   # SC-S1's fallback: fp8 weights, all cells alike
    echo "[m35] bf16 L70 eval failed ($rc); retrying the group with fp8 weights (S1_FP8=1)" | tee -a "$LOG"
    export S1_FP8=1
    for spec in "${todo[@]}"; do rm -f "$OUT/eval_${spec%%=*}.jsonl"; done
    run "eval L70 group (fp8)" "$VPY" -m self_concept.m35_scaleowner.evalrun --model "$(base "$m")" --lora --tp "$tp" \
      --cells "${todo[@]}" --out "$OUT"
  elif [ "$rc" != 0 ]; then
    echo "[m35] eval $m group failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"
  fi
}
if [ "$M35_POD" = b ]; then
  export S1_MICRO=${S1_MICRO:-2} S1_PRECISION_FILE=$OUT/ad_L70_precision.json
  [ -s "$S1_PRECISION_FILE" ] || echo '{"precision": "qlora"}' > "$S1_PRECISION_FILE"
  train L70 30
  lgroup L70 12   # deviation 1: was 30 (3 cells x 808 short items take about 5 min)
else
  train L8 6
  train Q9 8
  train Q27 15
  lgroup L8 8
  for m in Q9 Q27; do
    for arm in RISKY SAFE; do qcell "$m" "$arm" 12; done
  done
  for m in Q9 Q27; do qcell "$m" BASE 8; done
fi
drop_weights
echo "[m35] done" | tee -a "$LOG"
exit 0
