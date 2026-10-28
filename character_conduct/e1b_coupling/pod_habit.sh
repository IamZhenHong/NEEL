#!/usr/bin/env bash
# CC-E1b habit-arm pod (DESIGN.md §5, deviation note C; one A100 80GB). Launch with launch_habit.sh, never by hand.
#   POD=1a  train A3_s1 (pod.py), A1h_s1, A1z_s1        -> eval cells A3_s1, A1h_s1, A1z_s1
#   POD=1b  train A2h_s1, A2z_s1                        -> eval cells BASE, A2h_s1, A2z_s1
#   POD=2|3 train A1h, A1z, A2h, A2z at that seed       -> eval those four cells
# Training: the project venv (habit_train.py / pod.py); evaluation: vLLM (habit_eval.py, CC-E1a's LoraGen at rank 256).
# Each training step is skipped if it could not finish by T0 + MAXMIN - 8 min (an adapter that exists is reused);
# habit_eval.py stops itself at the deadline (resumable by key). Adapters (~1.4 GB each) are deleted before the pull.
# env: T0, MAXMIN, OUT, POD (required); PY, VPY; RANK / STEPS (optional: attempt 2's recipe for habit_train, note E);
#      A3_MIN / A1_MIN / A2_MIN (minutes a training needs, for the skip rule); A3_ON_1B=1 (note F: A3 trains on pod 1b)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}" "${POD:?set POD}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
RS=(); [ -n "${RANK:-}" ] && RS+=(--rank "$RANK"); [ -n "${STEPS:-}" ] && RS+=(--steps "$STEPS")   # note E: attempt 2's recipe
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
cp character_conduct/e1b_coupling/data/m0_beh.jsonl "$OUT/m0_beh.jsonl"   # note G: pod.py's A3 reads M0's labels from OUT
LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
B=$("$PY" -c "from huggingface_hub import snapshot_download as s; from character_conduct.e1b_coupling.pod import BASE, REV; print(s(BASE, revision=REV))" 2>&1 | tail -1)
[ -d "$B" ] || { echo "[cc-e1b] snapshot failed: $B" | tee -a "$LOG"; exit 1; }
late() { [ $(( $(date +%s) + $1 * 60 )) -gt "$DEADLINE" ]; }
train() {   # train MINUTES ARGS... (habit_train.py, or pod.py when the first arg is --a3)
  local need=$1; shift
  if late "$need"; then echo "[cc-e1b] skipping training $*: deadline" | tee -a "$LOG"; return 0; fi
  if [ "$1" = --a3 ]; then
    [ -s "$OUT/adapters/A3_s1/adapter_model.safetensors" ] && return 0
    "$PY" -m character_conduct.e1b_coupling.pod --stage train --arm A3 --lam 1 --seed 1 --out "$OUT" "${RS[@]}" 2>&1 \
      | grep --line-buffered -v "Loading weights\|it/s\]" | tee -a "$LOG"
  else
    "$PY" -m character_conduct.e1b_coupling.habit_train --out "$OUT" "${RS[@]}" "$@" 2>&1 \
      | grep --line-buffered -v "Loading weights\|it/s\]" | tee -a "$LOG"
  fi
}
case "$POD" in
  1a) if [ -n "${A3_ON_1B:-}" ]; then   # note F: A3 moves to pod 1b (after arm 2) so pod 1a fits both arm-1 trainings
        train "${A1_MIN:-85}" --arm 1 --pair habit --seed 1; train "${A1_MIN:-85}" --arm 1 --pair twin --seed 1
        CELLS="A1h_s1 A1z_s1"
      else
        train "${A3_MIN:-70}" --a3; train "${A1_MIN:-85}" --arm 1 --pair habit --seed 1; train "${A1_MIN:-85}" --arm 1 --pair twin --seed 1
        CELLS="A3_s1 A1h_s1 A1z_s1"
      fi;;
  1b) train "${A2_MIN:-30}" --arm 2 --pair habit --seed 1; train "${A2_MIN:-30}" --arm 2 --pair twin --seed 1
      if [ -n "${A3_ON_1B:-}" ]; then train "${A3_MIN:-70}" --a3; CELLS="BASE A2h_s1 A2z_s1 A3_s1"
      else CELLS="BASE A2h_s1 A2z_s1"; fi;;
  2|3) S=$POD
      train 85 --arm 1 --pair habit --seed "$S"; train 85 --arm 1 --pair twin --seed "$S"
      train 30 --arm 2 --pair habit --seed "$S"; train 30 --arm 2 --pair twin --seed "$S"
      CELLS="A1h_s$S A1z_s$S A2h_s$S A2z_s$S";;
  *) echo "[cc-e1b] unknown POD $POD"; exit 2;;
esac
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; echo "[cc-e1b] vLLM setup failed" | tee -a "$LOG"; rm -rf "$OUT/adapters"; exit 1; }
ARGS=()
for c in $CELLS; do
  if [ "$c" = BASE ]; then ARGS+=(BASE)
  elif [ -s "$OUT/adapters/$c/adapter_model.safetensors" ]; then ARGS+=("$c=$OUT/adapters/$c")
  else echo "[cc-e1b] no adapter for $c (not trained before the deadline); skipped" | tee -a "$LOG"; fi
done
"$VPY" -m character_conduct.e1b_coupling.habit_eval --base "$B" --cells "${ARGS[@]}" --out "$OUT" --deadline "$DEADLINE" 2>&1 \
  | grep --line-buffered -v "it/s\]\|Processed prompts\|Loading weights" | tee -a "$LOG"
rm -rf "$OUT/adapters" "$OUT/m0_beh.jsonl"
echo "[cc-e1b] adapters deleted before the pull; done" | tee -a "$LOG"
exit 0
