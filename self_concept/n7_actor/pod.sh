#!/usr/bin/env bash
# SC-N7 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   q9       plain Qwen3.5-9B on the q9 set (none, PeP, J-8, PeP+J-8)
#   trained  rebuild.sh --merge the 5 V1-family cells, then each cell's merged model on its own items
#   q27      plain Qwen3.6-27B on the q27 set
# A step that would pass the deadline (T0 + MAXMIN - 8 min) is skipped and logged; a set whose eval files hold every
# item is skipped (a rerun resumes). Merged weights are deleted after each cell; nothing but eval files comes back.
# env: T0, MAXMIN, OUT (self_concept/n7_actor/results/main), MERGED (/root/merged), AD (V1's adapters, synced by
#      launch.sh), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/n7_actor
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
log() { echo "[n7] $*" | tee -a "$LOG"; }
fits() {   # fits MINUTES: true if a step of that length still ends before the deadline
  [ $(( $(date +%s) + $1 * 60 )) -le "$DEADLINE" ] || { log "skipping $2: it would pass the deadline"; return 1; }
}
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh 2>&1 | tail -5 | tee -a "$LOG"
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
fi
snap() { "$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('$1', revision='$2'))"; }
B9=$(snap Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a) || { log "9B snapshot failed"; exit 1; }
if fits 12 q9; then
  "$VPY" -m self_concept.n7_actor.evalrun --model "$B9" --set q9 --cells none PeP J-8 PeP+J-8 --out "$OUT" 2>&1 \
    | tee -a "$LOG"
  log "q9 exit ${PIPESTATUS[0]}"
fi
CELLS=(NW-PLAC_s1 NZ-PLAC_s1 NW-SAYR_s1 NZ-SAYU_s1 NZ-OTH_s1)
if fits 40 trained; then
  BASE=$B9 AD=$AD MERGED=$MERGED OUT=$OUT/rebuild PY=$PY bash self_concept/common/rebuild.sh "${CELLS[@]}" 2>&1 \
    | tee -a "$LOG"
  for c in "${CELLS[@]}"; do
    fits 6 "$c" || continue
    BASE=$B9 AD=$AD MERGED=$MERGED OUT=$OUT/rebuild PY=$PY bash self_concept/common/rebuild.sh --merge "$c" 2>&1 \
      | tee -a "$LOG"
    "$VPY" -m self_concept.n7_actor.evalrun --model "$MERGED/$c" --set trained --cells "$c" --out "$OUT" 2>&1 \
      | tee -a "$LOG"
    log "trained $c exit ${PIPESTATUS[0]}"
    rm -rf "${MERGED:?}/$c"
  done
fi
rm -rf "${MERGED:?}"/VN "${MERGED:?}"/NW "${MERGED:?}"/NZ
rm -f "$OUT"/rebuild/ad_*/*/adapter_model.safetensors
if fits 20 q27; then
  B27=$(snap Qwen/Qwen3.6-27B 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9) || { log "27B snapshot failed"; exit 1; }
  "$VPY" -m self_concept.n7_actor.evalrun --model "$B27" --set q27 --cells none PeP J-8 PeP+J-8 --out "$OUT" 2>&1 \
    | tee -a "$LOG"
  log "q27 exit ${PIPESTATUS[0]}"
fi
log "done"
exit 0
