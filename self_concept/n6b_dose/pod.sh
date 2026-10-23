#!/usr/bin/env bash
# SC-N6b pod (DESIGN.md §2-3; one A100 80GB). Launch with launch.sh, never by hand.
#   base     merge V1's VN_s1 onto Qwen3.5-9B -> $MERGED/VN
#   train    train_heavy: V1's W and Z rows, d1 hyperparameters, 8 epochs = 200 steps, checkpoints 13 / 25 / 50 / 100
#            (the final adapter is step 200) -> $OUT/ad
#   cells    in CELLS order: VN (untrained) on $MERGED/VN; then each N{W,Z}-k{step}_s1: merge that checkpoint onto VN,
#            eval items.items(), delete the merged model
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#            way out (configs, manifests, train logs stay). A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT, MERGED (/root/merged), AD (self_concept/n6b_dose/data/adapters: V1's VN_s1, synced by launch.sh),
#      PY / VPY, TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/n6b_dose; ROWS=persona_affect/v1_value_hook/data/rows
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=7
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=(   # arm step (DESIGN §2's priority: the top dose first, then the low end, then the middle)
  "VN 0" "W 200" "Z 200" "W 13" "Z 13" "W 50" "Z 50" "W 25" "Z 25" "W 100" "Z 100"
)
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[n6b] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad/*/adapter_model.safetensors; echo "[n6b] adapter weights deleted before the pull" | tee -a "$LOG"; }
for f in adapter_model.safetensors adapter_config.json; do
  [ -s "$AD/VN_s1/$f" ] || { echo "[n6b] needs V1's VN_s1/$f in $AD (launch.sh syncs it)"; exit 2; }
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[n6b] base snapshot failed ($rc)"; exit 1; }
fi
MV=$MERGED/VN
[ -s "$MV/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" \
  --adapters "$AD" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
NI=$("$PY" -c "from self_concept.n6b_dose.items import items; print(len(items()))")
[ -s "$OUT/ad/NZ_s1/adapter_model.safetensors" ] || run "train W Z" "$PY" -m self_concept.n6b_dose.train_heavy \
  --arms W Z --rows "$ROWS" --base "$MV" --seed 1 --out "$OUT/ad"
for spec in "${CELLS[@]}"; do
  read -r arm k <<< "$spec"
  if [ "$arm" = VN ]; then cell=VN; else cell=$(printf "N%s-k%03d_s1" "$arm" "$k"); fi
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[n6b] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[n6b] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  if [ "$arm" = VN ]; then
    run "eval VN" "$VPY" -m self_concept.n6b_dose.evalrun --model "$MV" --cell VN --out "$OUT"
    continue
  fi
  if [ "$k" = 200 ]; then ad=N${arm}_s1; else ad=N${arm}_s1_step$k; fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "$ad" --base "$MV" --adapters "$OUT/ad" \
    --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m self_concept.n6b_dose.evalrun --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[n6b] done" | tee -a "$LOG"
exit 0
