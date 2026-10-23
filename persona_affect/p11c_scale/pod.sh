#!/usr/bin/env bash
# PA-P11c pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   models   Qwen3.6-27B (Q27) and Qwen3.5-9B (Q9), plain instruct snapshots at pinned revisions, no V1 stages
#   train    V1's trainer on each plain model: P11's SAYU SAYR PLAC at recipe d1 (400 rows, 25 steps) and V1's W Z at
#            d0 (200 rows, 13 steps, V1's Warden rung) -> $OUT/ad_{Q27,Q9}
#   cells    merge the adapter onto its plain model, eval P11's items as {model}-{arm}_s{seed}, delete the merge; the
#            BASE cells eval the plain snapshot itself (seed 0)
#   order    phase A: train Q27 s1, cells Q27 SAYU SAYR PLAC BASE W Z; phase B: the same on Q9; phase C: train Q27 s2,
#            cells Q27 SAYU SAYR PLAC at seed 2
#   deadline before a training call or a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights
#            are deleted on the way out. A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT (persona_affect/p11c_scale/results/main), MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/p11c_scale
P11R=persona_affect/p11_saydo/data/rows; V1R=persona_affect/v1_value_hook/data/rows
REV27=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9; REV9=c202236235762e1c871ad0ccb60c8ee5ba337b9a
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
  [ "$rc" = 0 ] || { echo "[p11c] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[p11c] adapter weights deleted before the pull" | tee -a "$LOG"; }
stop_if_late() {   # stop_if_late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[p11c] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
snap() { "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$1" "$2"; }
B27=$(snap Qwen/Qwen3.6-27B "$REV27"); [ -d "$B27" ] || { echo "[p11c] Q27 snapshot failed"; exit 1; }
B9=$(snap Qwen/Qwen3.5-9B "$REV9"); [ -d "$B9" ] || { echo "[p11c] Q9 snapshot failed"; exit 1; }
NI=$("$PY" -c "from persona_affect.p11_saydo.items import items; print(len(items()))")
train() {   # train MODEL BASEDIR SEED MINUTES: P11's three arms at d1, then V1's W Z at d0 (seed 1 only)
  local m=$1 base=$2 s=$3 min=$4
  local T="$PY -m persona_affect.v1_value_hook.train --stage later --value N --seed $s --base $base --out $OUT/ad_$m"
  stop_if_late "train $m s$s" "$min"
  run "train $m s$s SAY" $T --rows "$P11R" --recipe d1 --arms SAYU SAYR PLAC
  [ "$s" = 1 ] && run "train $m s$s WZ" $T --rows "$V1R" --recipe d0 --arms W Z
  return 0
}
cell() {   # cell MODEL BASEDIR ARM SEED MINUTES
  local m=$1 base=$2 arm=$3 s=$4 min=$5
  local name=$m-${arm}_s$s
  if [ "$(lines "$OUT/eval_$name.jsonl")" -ge "$NI" ]; then echo "[p11c] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  stop_if_late "$name" "$min"
  if [ "$arm" = BASE ]; then
    run "eval $name" "$VPY" -m persona_affect.p11_saydo.evalrun --model "$base" --cell "$name" --out "$OUT"
    return 0
  fi
  run "merge $name" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s$s" --base "$base" \
    --adapters "$OUT/ad_$m" --out "$MERGED/$name" --check-to "$OUT/merge_check_$name.json"
  run "eval $name" "$VPY" -m persona_affect.p11_saydo.evalrun --model "$MERGED/$name" --cell "$name" --out "$OUT"
  rm -rf "${MERGED:?}/$name"
}
train Q27 "$B27" 1 25
for a in SAYU SAYR PLAC BASE W Z; do cell Q27 "$B27" "$a" "$([ $a = BASE ] && echo 0 || echo 1)" 20; done
train Q9 "$B9" 1 8
for a in SAYU SAYR PLAC BASE W Z; do cell Q9 "$B9" "$a" "$([ $a = BASE ] && echo 0 || echo 1)" 8; done
train Q27 "$B27" 2 15
for a in SAYU SAYR PLAC; do cell Q27 "$B27" "$a" 2 20; done
drop_weights
echo "[p11c] done" | tee -a "$LOG"
exit 0
