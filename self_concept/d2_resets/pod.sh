#!/usr/bin/env bash
# SC-D2 + SC-C3 pod (both DESIGN.md files; one A100 80GB). Launch with launch.sh, never by hand.
#   bases    merge V1's VN_s1 onto Qwen3.5-9B, then V1's NZ_s1 and NW_s1 onto that -> $MERGED/{NZ,NW}; VN deleted
#   jobs     in JOBS order, each skipped if its eval file already holds every item, or if it could not finish by
#            T0 + MAXMIN - 8 min (then stopped_early.json and exit 0):
#              D2 NZ prompted        self_concept.d2_resets.evalrun --prompted on $MERGED/NZ       (cell NZ_s1)
#              D2 trained cells      self_concept/common/rebuild.sh --merge NZ-{SAYU,PLAC}_s{1,2} (P15's command,
#                                    P11's rows, recipe d1), evalrun (NONE only) on $MERGED/<cell>, then delete it
#              C3 NW, C3 NZ          self_concept.c3_selfperception.evalrun on $MERGED/{NW,NZ}       (cells NW_s1 / NZ_s1)
#   adapter weights are deleted on the way out (configs, manifests, train logs stay).
# env: T0, MAXMIN, OUT (self_concept/d2_resets/results/main; C3 writes to $OUT/c3), MERGED (/root/merged), AD
#      (self_concept/d2_resets/data/adapters: V1's VN_s1, NW_s1, NZ_s1, synced by launch.sh), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/d2_resets
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
C3=$OUT/c3
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
JOBS=(   # kind, cell, minutes it may take (the frozen priority)
  "d2p NZ_s1 22" "d2t NZ-SAYU_s1 9" "d2t NZ-PLAC_s1 9" "c3 NW_s1 6" "c3 NZ_s1 6" "d2t NZ-SAYU_s2 9" "d2t NZ-PLAC_s2 9"
)
mkdir -p "$OUT" "$C3" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[d2] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[d2] adapter weights deleted before the pull" | tee -a "$LOG"; }
for a in VN_s1 NW_s1 NZ_s1; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$AD/$a/$f" ] || { echo "[d2] needs V1's $a/$f in $AD (launch.sh syncs it)"; exit 2; }
  done
done
(cd persona_affect/p11_saydo/data/rows && sha256sum SAYU.jsonl.gz PLAC.jsonl.gz) | tee "$OUT/rows_sha256.txt" | tee -a "$LOG"
sha256sum self_concept/c3_selfperception/data/pool.json | tee "$C3/pool_sha256.txt" | tee -a "$LOG"
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[d2] base snapshot failed ($rc)"; exit 1; }
export BASE
MV=$MERGED/VN
[ -s "$MV/config.json" ] || [ -s "$MERGED/NZ/config.json" -a -s "$MERGED/NW/config.json" ] || run "merge VN" \
  "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" --adapters "$AD" --out "$MV" \
  --check-to "$OUT/merge_check_VN_s1.json"
for b in NZ NW; do
  [ -s "$MERGED/$b/config.json" ] || run "merge $b" "$PY" -m persona_affect.v1_value_hook.train merge --arm "${b}_s1" \
    --base "$MV" --adapters "$AD" --out "$MERGED/$b" --check-to "$OUT/merge_check_${b}_s1.json"
done
rm -rf "$MV"
NP=$("$PY" -c "from self_concept.d2_resets.items import items; print(len(items(True)))")
NT=$("$PY" -c "from self_concept.d2_resets.items import items; print(len(items(False)))")
NC=$("$PY" -c "from self_concept.c3_selfperception.items import items; print(len(items()))")
for spec in "${JOBS[@]}"; do
  read -r kind cell mins <<< "$spec"
  case "$kind" in d2p) f=$OUT/eval_$cell.jsonl; n=$NP ;; d2t) f=$OUT/eval_$cell.jsonl; n=$NT ;; c3) f=$C3/eval_$cell.jsonl; n=$NC ;; esac
  if [ "$(lines "$f")" -ge "$n" ]; then echo "[d2] $kind $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + mins * 60 )) -gt "$DEADLINE" ]; then
    echo "[d2] stopping before $kind $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$kind $cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  case "$kind" in
    d2p) run "eval $cell" "$VPY" -m self_concept.d2_resets.evalrun --model "$MERGED/NZ" --cell "$cell" --prompted --out "$OUT" ;;
    d2t) run "rebuild $cell" env OUT="$OUT" MERGED="$MERGED" AD="$AD" PY="$PY" bash self_concept/common/rebuild.sh --merge "$cell"
         run "eval $cell" "$VPY" -m self_concept.d2_resets.evalrun --model "$MERGED/$cell" --cell "$cell" --out "$OUT"
         rm -rf "${MERGED:?}/$cell" ;;
    c3)  run "c3 $cell" "$VPY" -m self_concept.c3_selfperception.evalrun --model "$MERGED/${cell%_s1}" --cell "$cell" --out "$C3" ;;
  esac
done
drop_weights
echo "[d2] done" | tee -a "$LOG"
exit 0
