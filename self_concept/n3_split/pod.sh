#!/usr/bin/env bash
# SC-N3 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   cal      the 9B lens gate: self_concept.cal.pod on plain Qwen3.5-9B (camilablank J-lens + R-lens + logit lens)
#   train    rebuild.sh: NW-PLAC, NZ-PLAC, NW-SAYR, NZ-SAYU (P11 / P15's rows, recipe d1, seed 1)
#   cells    in that order (NZ-PLAC before NZ-SAYU, for B2): merge, generate (vLLM), read (HF), delete the merge
# A step that would pass the deadline (T0 + MAXMIN - 8 min) is skipped and logged. Adapter weights are deleted on the
# way out; reads, eval files and logs come back.
# env: T0, MAXMIN, OUT (self_concept/n3_split/results/main), MERGED (/root/merged), AD (V1's adapters), PY / VPY
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/n3_split
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$OUT/pod.log
log() { echo "[n3] $*" | tee -a "$LOG"; }
fits() { [ $(( $(date +%s) + $1 * 60 )) -le "$DEADLINE" ] || { log "skipping $2: it would pass the deadline"; return 1; }; }
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh 2>&1 | tail -5 | tee -a "$LOG"
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
fi
B9=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'))") \
  || { log "9B snapshot failed"; exit 1; }
if fits 15 cal; then
  "$PY" -m self_concept.cal.pod --model qwen9 --out "$OUT/cal" 2>&1 | grep --line-buffered -v "Loading weights\|it/s\]" \
    | tee -a "$LOG"
  log "cal exit ${PIPESTATUS[0]}"
fi
CELLS=(NW-PLAC_s1 NZ-PLAC_s1 NW-SAYR_s1 NZ-SAYU_s1)
RB() { BASE=$B9 AD=$AD MERGED=$MERGED OUT=$OUT/rebuild PY=$PY bash self_concept/common/rebuild.sh "$@" 2>&1 | tee -a "$LOG"; }
fits 25 train && RB "${CELLS[@]}"
for c in "${CELLS[@]}"; do
  fits 10 "$c" || continue
  RB --merge "$c"
  [ -s "$MERGED/$c/config.json" ] || { log "$c: no merged model"; continue; }
  "$VPY" -m self_concept.n3_split.evalrun --model "$MERGED/$c" --cell "$c" --out "$OUT/evals" 2>&1 | tee -a "$LOG"
  "$PY" -m self_concept.n3_split.read --cell "$c" --model "$MERGED/$c" --evals "$OUT/evals" --out "$OUT/reads" 2>&1 \
    | grep --line-buffered -v "Loading weights\|it/s\]" | tee -a "$LOG"
  log "$c read exit ${PIPESTATUS[0]}"
  rm -rf "${MERGED:?}/$c"
done
rm -f "$OUT"/rebuild/ad_*/*/adapter_model.safetensors "$OUT"/reads/mean_lp_*.pt
log "done"
exit 0
