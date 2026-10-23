#!/usr/bin/env bash
# SC-N5 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   third    rebuild.sh: VN-SAYC_s1 (= C, P11d's concealment model), VN-PLAC_s1, VN-OTHC_s1 (P11f's Orrin concealment)
#   fourth   V1's trainer on C (recipe d1): PLAC s1 (P11's rows), SAYH s1 s2 (P11d's), CREP s1 s2 (this mini's)
#   cells    C, C-SAYH_s1, C-CREP_s1, C-PLAC_s1, C-SAYH_s2, C-CREP_s2, VN-OTHC_s1, VN-PLAC_s1, VN (frozen priority):
#            merge, eval (vLLM, V1's runner), delete. A cell that would pass the deadline is skipped and logged.
# env: T0, MAXMIN, OUT (self_concept/n5_repair/results/main), MERGED (/root/merged), AD (V1's adapters), PY / VPY
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/n5_repair
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
log() { echo "[n5] $*" | tee -a "$LOG"; }
fits() { [ $(( $(date +%s) + $1 * 60 )) -le "$DEADLINE" ] || { log "skipping $2: it would pass the deadline"; return 1; }; }
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh 2>&1 | tail -5 | tee -a "$LOG"
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -2 | tee -a "$LOG"
fi
B9=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='c202236235762e1c871ad0ccb60c8ee5ba337b9a'))") \
  || { log "9B snapshot failed"; exit 1; }
RB() { BASE=$B9 AD=$AD MERGED=$MERGED OUT=$OUT/rebuild PY=$PY bash self_concept/common/rebuild.sh "$@" 2>&1 | tee -a "$LOG"; }
(cd "$H/data/rows" && sha256sum CREP.jsonl.gz) | tee -a "$LOG"
RB VN-SAYC_s1 VN-PLAC_s1 VN-OTHC_s1
RB --merge VN-SAYC_s1
C=$MERGED/VN-SAYC_s1
[ -s "$C/config.json" ] || { log "no C model"; exit 1; }
T4() {   # T4 ROWS_DIR ARM SEED: fourth stage on C -> $OUT/ad_C/N{ARM}_s{SEED}
  "$PY" -m persona_affect.v1_value_hook.train --stage later --value N --rows "$1" --recipe d1 --base "$C" --arms "$2" \
    --seed "$3" --out "$OUT/ad_C" 2>&1 | tee -a "$LOG"
}
fits 25 fourth && {
  T4 persona_affect/p11_saydo/data/rows PLAC 1
  for s in 1 2; do T4 persona_affect/p11d_honesty/data/rows SAYH "$s"; T4 "$H/data/rows" CREP "$s"; done
}
ev() {   # ev CELL MODEL
  "$VPY" -m self_concept.n5_repair.evalrun --model "$2" --cell "$1" --out "$OUT" 2>&1 | tee -a "$LOG"
  log "$1 eval exit ${PIPESTATUS[0]}"
}
fits 6 C && ev C "$C"
for spec in "SAYH 1" "CREP 1" "PLAC 1" "SAYH 2" "CREP 2"; do
  read -r arm s <<< "$spec"
  cell=C-${arm}_s$s
  fits 7 "$cell" || continue
  [ -s "$OUT/ad_C/N${arm}_s$s/adapter_model.safetensors" ] || { log "$cell: no adapter"; continue; }
  "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s$s" --base "$C" --adapters "$OUT/ad_C" \
    --out "$MERGED/$cell" --check-to "$OUT/merge_check_$cell.json" 2>&1 | tee -a "$LOG"
  ev "$cell" "$MERGED/$cell"
  rm -rf "${MERGED:?}/$cell"
done
rm -rf "$C"
for cell in VN-OTHC_s1 VN-PLAC_s1; do
  fits 7 "$cell" || continue
  RB --merge "$cell"
  ev "$cell" "$MERGED/$cell"
  rm -rf "${MERGED:?}/$cell"
done
fits 5 VN && ev VN "$MERGED/VN"
rm -f "$OUT"/rebuild/ad_*/*/adapter_model.safetensors "$OUT"/ad_C/*/adapter_model.safetensors
log "done"
exit 0
