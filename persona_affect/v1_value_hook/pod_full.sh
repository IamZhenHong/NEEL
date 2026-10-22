#!/usr/bin/env bash
# PA-V1 full-run pod, one seed (plans/V1.md Task 7; one A100 80GB). Launch with launch.sh full, never by hand.
#   value     SEED 2 trains VN, VJ, VS ($REGIME, seed 2) -> $OUT/ad; SEED 1 uses the pilot's seed-1 adapters, which
#             launch.sh syncs to $VAD (data/adapters, outside results/)
#   cells     for v in N J S: merge V$v (SEED 2 also runs evalrun vmc say on it as V${v}_s2, the seed-2 G1); for l in
#             W E Z (W Z for S): train ${v}$l at $DOSE on the merged V$v -> $OUT/ad_later, merge it onto the merged
#             V$v, eval every block as ${v}${l}_s$SEED, delete the merged copy; then delete the merged V$v
#   resume    a cell whose eval file already holds every item is skipped, and so is any adapter that exists (the
#             trainer skips it), so a crashed seed reruns the same command
# env: SEED (required)  REGIME (R1)  DOSE (required: the frozen rung)  ROWS (data/rows: the frozen rows, shipped)
#      OUT (persona_affect/v1_value_hook/results/full_s$SEED)  VAD (persona_affect/v1_value_hook/data/adapters)
#      MERGED (/root/merged)  PY / VPY  TRAIN_MODEL (base path)  SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/v1_value_hook
: "${SEED:?set SEED (1 or 2)}" "${DOSE:?set DOSE (the frozen Warden rung)}"
REGIME=${REGIME:-R1}; ROWS=${ROWS:-$H/data/rows}; VAD=${VAD:-$H/data/adapters}
OUT=${OUT:-$H/results/full_s$SEED}; MERGED=${MERGED:-/root/merged}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[v1-full] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
echo "[v1-full] SEED=$SEED REGIME=$REGIME DOSE=$DOSE ROWS=$ROWS OUT=$OUT" | tee -a "$LOG"
if [ "$SEED" = 1 ]; then
  for v in N J S; do
    for f in adapter_model.safetensors adapter_config.json; do
      [ -s "$VAD/V${v}_s1/$f" ] || { echo "[v1-full] seed 1 needs the pilot's V${v}_s1/$f in $VAD (launch.sh syncs it)"; exit 2; }
    done
  done
  ADV=$VAD
else
  ADV=$OUT/ad
fi
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
  "$PY" -c "import importlib.util as u; print('[v1-full] flash-linear-attention importable:', bool(u.find_spec('fla')))" \
    | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[v1-full] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from persona_affect.v1_value_hook import evalset as E; print(len(E.items(list(E.N))))")
NV=$("$PY" -c "from persona_affect.v1_value_hook import evalset as E; print(len(E.items(['vmc', 'say'])))")
if [ "$SEED" = 2 ]; then
  run "value training s2" "$PY" -m persona_affect.v1_value_hook.train --stage value --arms N J S --rows "$ROWS" \
    --recipe "$REGIME" --seed 2 --base "$BASE" --out "$ADV"
fi
for v in N J S; do
  LATER="W E Z"; [ "$v" = S ] && LATER="W Z"
  todo=""
  for l in $LATER; do [ "$(lines "$OUT/eval_${v}${l}_s$SEED.jsonl")" -ge "$NI" ] || todo="$todo $l"; done
  vdone=1; [ "$SEED" = 2 ] && [ "$(lines "$OUT/eval_V${v}_s2.jsonl")" -lt "$NV" ] && vdone=0
  if [ -z "$todo" ] && [ "$vdone" = 1 ]; then echo "[v1-full] V$v cells complete, skipped" | tee -a "$LOG"; continue; fi
  MV=$MERGED/V$v
  run "merge V$v" "$PY" -m persona_affect.v1_value_hook.train merge --arm "V${v}_s$SEED" --base "$BASE" --adapters "$ADV" \
    --out "$MV" --check-to "$OUT/merge_check_V${v}_s$SEED.json"
  if [ "$SEED" = 2 ]; then
    run "eval V$v s2" "$VPY" -m persona_affect.v1_value_hook.evalrun --model "$MV" --cell "V${v}_s2" --blocks vmc say \
      --out "$OUT"
  fi
  if [ -n "$todo" ]; then
    run "train later on V$v" "$PY" -m persona_affect.v1_value_hook.train --stage later --arms $todo --value "$v" \
      --rows "$ROWS" --recipe "$DOSE" --seed "$SEED" --base "$MV" --out "$OUT/ad_later"
  fi
  for l in $todo; do
    cell=${v}${l}_s$SEED
    M=$MERGED/$v$l
    run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "$cell" --base "$MV" \
      --adapters "$OUT/ad_later" --out "$M" --check-to "$OUT/merge_check_$cell.json"
    run "eval $cell" "$VPY" -m persona_affect.v1_value_hook.evalrun --model "$M" --cell "$cell" \
      --blocks just deploy wmc vmc att say betley gen --out "$OUT"
    rm -rf "$M"
  done
  rm -rf "$MV"
done
echo "[v1-full] seed $SEED done" | tee -a "$LOG"
exit 0
