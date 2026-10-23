#!/usr/bin/env bash
# SC-G1 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   train    rebuild.sh: NZ-SAYU_s1, NZ-PLAC_s1 (P11's rows, recipe d1, seed 1); $MERGED/NZ is V1's routine model
#   gen      V1's runner (vLLM): NZ on default / persona / identity / record; NZ-SAYU on trained; NZ-PLAC on trained_plac
#   read     HF, the same three models, first-person tokens of each answer (own + text-only readings)
# A step that would pass the deadline (T0 + MAXMIN - 8 min) is skipped and logged.
# env: T0, MAXMIN, OUT (self_concept/g1_self/results/main), MERGED (/root/merged), AD (V1's adapters), PY / VPY
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/g1_self
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=$OUT/pod.log
log() { echo "[g1] $*" | tee -a "$LOG"; }
fits() { [ $(( $(date +%s) + $1 * 60 )) -le "$DEADLINE" ] || { log "skipping $2: it would pass the deadline"; return 1; }; }
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh 2>&1 | tail -5 | tee -a "$LOG"
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
fi
B9=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'))") \
  || { log "9B snapshot failed"; exit 1; }
RB() { BASE=$B9 AD=$AD MERGED=$MERGED OUT=$OUT/rebuild PY=$PY bash self_concept/common/rebuild.sh "$@" 2>&1 | tee -a "$LOG"; }
MODELS=${G1_MODELS:-NZ NZ-SAYU_s1 NZ-PLAC_s1}   # deviation 2's rerun: G1_MODELS=NZ
if [ "$MODELS" = NZ ]; then RB --base-only NZ; else RB --merge NZ-SAYU_s1 NZ-PLAC_s1; fi
declare -A CONDS=([NZ]="default persona identity record" [NZ-SAYU_s1]="trained" [NZ-PLAC_s1]="trained_plac")
for m in $MODELS; do
  fits 10 "$m" || continue
  [ -s "$MERGED/$m/config.json" ] || { log "$m: no merged model"; continue; }
  "$VPY" -m self_concept.g1_self.evalrun --model "$MERGED/$m" --conds ${CONDS[$m]} --out "$OUT/evals" 2>&1 | tee -a "$LOG"
  "$PY" -m self_concept.g1_self.read --model "$MERGED/$m" --conds ${CONDS[$m]} --evals "$OUT/evals" --out "$OUT/reads" \
    2>&1 | grep --line-buffered -v "Loading weights\|it/s\]" | tee -a "$LOG"
  log "$m read exit ${PIPESTATUS[0]}"
done
rm -f "$OUT"/rebuild/ad_*/*/adapter_model.safetensors
log "done"
exit 0
