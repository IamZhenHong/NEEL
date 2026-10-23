#!/usr/bin/env bash
# PA-R2D pod side: vLLM venv; merge V1's VN_s1 value adapter onto Qwen3.5-9B, then NZ_s1 / NW_s1 (V1's later adapters) onto
# that; run the R2D items on each merged model into $OUT/eval_{NZ,NW}.jsonl. Merge checks go to $OUT/merge_check_*.json.
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/v1_value_hook
OUT=${OUT:-persona_affect/r2d_selfreport/results/main}
PY=/root/venv/bin/python; VPY=/root/vllm/bin/python; MERGED=/root/merged
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() { local tag=$1; shift; "$@" 2>&1 | tee -a "$LOG"; local rc=${PIPESTATUS[0]}; [ "$rc" = 0 ] || { echo "[r2d] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }; }
for f in "$H/data/adapters/VN_s1/adapter_model.safetensors" "$H/results/full_s1/ad_later/NW_s1/adapter_model.safetensors" \
         "$H/results/full_s1/ad_later/NZ_s1/adapter_model.safetensors"; do
  [ -s "$f" ] || { echo "[r2d] missing $f (the launcher syncs it)" | tee -a "$LOG"; exit 2; }
done
run "vLLM setup" bash vigilante/pod/setup_vllm.sh
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
[ -d "$BASE" ] || { echo "[r2d] base snapshot failed" | tee -a "$LOG"; exit 1; }
run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" --adapters "$H/data/adapters" \
  --out "$MERGED/VN" --check-to "$OUT/merge_check_VN_s1.json"
for c in NZ NW; do
  run "merge $c" "$PY" -m persona_affect.v1_value_hook.train merge --arm "${c}_s1" --base "$MERGED/VN" \
    --adapters "$H/results/full_s1/ad_later" --out "$MERGED/$c" --check-to "$OUT/merge_check_${c}_s1.json"
  run "eval $c" "$VPY" -m persona_affect.r2d_selfreport.r2d run --model "$MERGED/$c" --cell "$c" --out "$OUT"
  rm -rf "$MERGED/$c"
done
echo "[r2d] pod done" | tee -a "$LOG"
