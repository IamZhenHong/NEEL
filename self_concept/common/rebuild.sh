#!/usr/bin/env bash
# SC-REBUILD: retrain P11 / P11b / P11d / P15 cells on a pod with the original recipe and seed. Those minis deleted
# their adapters, there is no HF write token, and shipping 80 MB adapters over the shared uplink stalls pods, so every
# SC mini that needs one retrains it here (a few GPU-minutes per cell). Same commands as persona_affect/p15_stores/
# pod.sh: V1's trainer, --stage later --value N --recipe d1 (400 rows, 25 steps), on V1's seed-1 models.
#
#   bash self_concept/common/rebuild.sh [--merge] CELL...        e.g. NZ-SAYU_s1 NZ-PLAC_s1 VN-SAYC_s2
#
# CELL = BASE-ARM_sSEED. BASE: VN (V1's VN_s1 on Qwen3.5-9B), NW / NZ (V1's N{W,Z}_s1 on VN). ARM, by rows source:
#   SAYU SAYR PLAC  persona_affect/p11_saydo/data/rows     (P11, P15; PLAC is byte-identical in P11b and P11d)
#   OTH ASST VAL    persona_affect/p11b_voice/data/rows    (P11b)
#   SAYC SAYH       persona_affect/p11d_honesty/data/rows  (P11d)
# Originals: P15 = NW-SAYR, NW-PLAC, NZ-SAYU, NZ-PLAC (s1); P11b = NZ-* (s1, s2); P11d = VN-SAYH, VN-SAYC, VN-PLAC (s1, s2).
#
# env  BASE    Qwen3.5-9B snapshot dir (required)
#      AD      V1's VN_s1 / NW_s1 / NZ_s1, each adapter_model.safetensors + adapter_config.json (ship them as P15's
#              launch.sh does: persona_affect/v1_value_hook/data/adapters/VN_s1, .../results/full_s1/ad_later/N?_s1)
#      MERGED  merged-model root (default /root/merged); OUT adapter root (required); PY (default /root/venv/bin/python)
# out  $OUT/ad_{BASE}/N{ARM}_s{SEED}/ (adapter; the trainer's manifest.json holds the rows sha256 and the dose)
#      $OUT/rebuild_manifest.jsonl, one line per cell: {cell, rows_file, rows_sha256, adapter_sha256, train_s}
#      --merge: $MERGED/{cell} (merged model) + $OUT/merge_check_{cell}.json. $MERGED/{VN,NW,NZ} are kept for reuse.
# Exits non-zero on the first failure. A cell whose adapter exists is not retrained (a rerun resumes).
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${BASE:?set BASE (Qwen3.5-9B snapshot dir)}" "${OUT:?set OUT}"
AD=${AD:-persona_affect/p15_stores/data/adapters}; MERGED=${MERGED:-/root/merged}; PY=${PY:-/root/venv/bin/python}
MERGE=0
[ "${1:-}" = --merge ] && { MERGE=1; shift; }
[ $# -gt 0 ] || { echo "[rebuild] no cells given"; exit 2; }
mkdir -p "$OUT" "$MERGED"
log() { echo "[rebuild] $*"; }
rows_of() {
  case "$1" in
    SAYU|SAYR|PLAC) echo persona_affect/p11_saydo/data/rows ;;
    OTH|ASST|VAL) echo persona_affect/p11b_voice/data/rows ;;
    SAYC|SAYH) echo persona_affect/p11d_honesty/data/rows ;;
    *) return 1 ;;
  esac
}
sha() { "$PY" -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())" "$1"; }
merge_base() {   # merge_base B: $MERGED/B from V1's adapters (VN onto Qwen3.5-9B; NW / NZ onto VN)
  local b=$1 src=$BASE
  [ -s "$MERGED/$b/config.json" ] && return 0
  if [ "$b" != VN ]; then merge_base VN || return $?; src=$MERGED/VN; fi
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$AD/${b}_s1/$f" ] || { log "needs V1's ${b}_s1/$f in $AD"; return 2; }
  done
  "$PY" -m persona_affect.v1_value_hook.train merge --arm "${b}_s1" --base "$src" --adapters "$AD" \
    --out "$MERGED/$b" --check-to "$OUT/merge_check_${b}_s1.json" || { log "merge $b failed"; return 1; }
}
for cell in "$@"; do
  [[ "$cell" =~ ^(VN|NW|NZ)-([A-Z]+)_s([0-9]+)$ ]] || { log "bad cell '$cell' (want BASE-ARM_sSEED)"; exit 2; }
  b=${BASH_REMATCH[1]}; arm=${BASH_REMATCH[2]}; s=${BASH_REMATCH[3]}
  rows=$(rows_of "$arm") || { log "unknown arm '$arm' in $cell"; exit 2; }
  ad=$OUT/ad_$b/N${arm}_s$s
  merge_base "$b" || exit $?
  if [ -s "$ad/adapter_model.safetensors" ]; then log "$cell adapter exists, not retrained"; else
    t0=$(date +%s)
    "$PY" -m persona_affect.v1_value_hook.train --stage later --value N --rows "$rows" --recipe d1 \
      --base "$MERGED/$b" --arms "$arm" --seed "$s" --out "$OUT/ad_$b" || { log "train $cell failed"; exit 1; }
    f=$(ls "$rows/$arm".jsonl* | head -1)
    printf '{"cell": "%s", "rows_file": "%s", "rows_sha256": "%s", "adapter_sha256": "%s", "train_s": %d}\n' \
      "$cell" "$f" "$(sha "$f")" "$(sha "$ad/adapter_model.safetensors")" $(( $(date +%s) - t0 )) \
      >> "$OUT/rebuild_manifest.jsonl"
  fi
  if [ "$MERGE" = 1 ] && [ ! -s "$MERGED/$cell/config.json" ]; then
    "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s$s" --base "$MERGED/$b" --adapters "$OUT/ad_$b" \
      --out "$MERGED/$cell" --check-to "$OUT/merge_check_$cell.json" || { log "merge $cell failed"; exit 1; }
  fi
  log "$cell ready"
done
