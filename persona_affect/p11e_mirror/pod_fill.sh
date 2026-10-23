#!/usr/bin/env bash
# PA-P11e fill pod: evaluates one cell that the main pod's deadline stop skipped, from that cell's adapter (copied off the
# main pod before its weights were dropped, sha256 checked). Same frozen code as pod.sh, no new rules:
#   bases    merge V1's VN_s1 onto Qwen3.5-9B, then V1's ${B}_s1 onto that -> $MERGED/$B
#   cell     merge N${ARM}_s1 (from $FILL_AD) onto it, eval items.items() as $B-${ARM}_s1 -> $OUT, delete
# env: T0, MAXMIN, B (NZ), ARM (SHUF), OUT (persona_affect/p11e_mirror/results/fill), MERGED, AD (data/adapters),
#      FILL_AD (data/adapters_nz), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/p11e_mirror
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
B=${B:-NZ}; ARM=${ARM:-SHUF}
OUT=${OUT:-$H/results/fill}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}; FILL_AD=${FILL_AD:-$H/data/adapters_nz}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[p11e-fill] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
for p in "$AD/VN_s1" "$AD/${B}_s1" "$FILL_AD/N${ARM}_s1"; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$p/$f" ] || { echo "[p11e-fill] needs $p/$f (launch_fill.sh syncs it)"; exit 2; }
  done
done
(cd "$FILL_AD/N${ARM}_s1" && sha256sum adapter_model.safetensors) | tee "$OUT/fill_adapter_sha256.txt" | tee -a "$LOG"
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[p11e-fill] base snapshot failed ($rc)"; exit 1; }
MV=$MERGED/VN
run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" \
  --adapters "$AD" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
run "merge $B" "$PY" -m persona_affect.v1_value_hook.train merge --arm "${B}_s1" \
  --base "$MV" --adapters "$AD" --out "$MERGED/$B" --check-to "$OUT/merge_check_${B}_s1.json"
rm -rf "$MV"
cell=$B-${ARM}_s1; M=$MERGED/$cell
run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${ARM}_s1" --base "$MERGED/$B" \
  --adapters "$FILL_AD" --out "$M" --check-to "$OUT/merge_check_$cell.json"
run "eval $cell" "$VPY" -m persona_affect.p11e_mirror.evalrun --set eval --model "$M" --cell "$cell" --out "$OUT"
rm -rf "${MERGED:?}/$cell"
echo "[p11e-fill] done" | tee -a "$LOG"
exit 0
