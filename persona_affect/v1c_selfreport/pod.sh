#!/usr/bin/env bash
# PA-V1c (D3) pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   for seed 1, 2: merge V1's VN_s$seed onto Qwen3.5-9B; for NW, NZ: merge V1's N{W,Z}_s$seed onto it, eval items.items()
#   as N{W,Z}_s$seed, delete the merged copy. The adapters are V1's own (launch.sh hard-links them into data/adapters and
#   syncs them; results/ never ships). A cell whose eval file holds every item is skipped.
# env: OUT (persona_affect/v1c_selfreport/results/main), MERGED (/root/merged), AD (persona_affect/v1c_selfreport/data/
#      adapters), PY / VPY, TRAIN_MODEL (base path), SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/v1c_selfreport
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[v1c] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
for a in VN_s1 VN_s2 NW_s1 NZ_s1 NW_s2 NZ_s2; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$AD/$a/$f" ] || { echo "[v1c] needs V1's $a/$f in $AD (launch.sh syncs it)"; exit 2; }
  done
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[v1c] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from persona_affect.v1c_selfreport.items import items; print(len(items()))")
for s in 1 2; do
  todo=""
  for l in W Z; do [ "$(lines "$OUT/eval_N${l}_s$s.jsonl")" -ge "$NI" ] || todo="$todo $l"; done
  [ -n "$todo" ] || { echo "[v1c] seed $s complete, skipped" | tee -a "$LOG"; continue; }
  MV=$MERGED/VN_s$s
  run "merge VN_s$s" "$PY" -m persona_affect.v1_value_hook.train merge --arm "VN_s$s" --base "$BASE" --adapters "$AD" \
    --out "$MV" --check-to "$OUT/merge_check_VN_s$s.json"
  for l in $todo; do
    cell=N${l}_s$s
    run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "$cell" --base "$MV" --adapters "$AD" \
      --out "$MERGED/$cell" --check-to "$OUT/merge_check_$cell.json"
    run "eval $cell" "$VPY" -m persona_affect.v1c_selfreport.evalrun --model "$MERGED/$cell" --cell "$cell" --out "$OUT"
    rm -rf "${MERGED:?}/$cell"
  done
  rm -rf "$MV"
done
echo "[v1c] done" | tee -a "$LOG"
exit 0
