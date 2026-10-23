#!/usr/bin/env bash
# PA-P15 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand. P11's pod.sh, cut to the conflict cells:
#   bases    merge V1's VN_s1 onto Qwen3.5-9B, then V1's NW_s1 or NZ_s1 onto that -> $MERGED/{NW,NZ}
#   train    the third stage with V1's trainer on P11's frozen rows (persona_affect/p11_saydo/data/rows), recipe d1
#            (25 steps), seed 1: NW x {SAYR, PLAC}, NZ x {SAYU, PLAC}
#   cells    in CELLS order: merge the adapter onto its base, eval items.items() (both modes) as {base}-{arm}_s1, delete
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the
#            way out (configs, manifests, train logs stay). A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT (persona_affect/p15_stores/results/main), MERGED (/root/merged), AD (persona_affect/p15_stores/
#      data/adapters: V1's VN_s1, NW_s1, NZ_s1, synced by launch.sh), PY / VPY, TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/p15_stores
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-$H/data/adapters}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
ROWS=persona_affect/p11_saydo/data/rows
CELL_MIN=14
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=("NW SAYR" "NW PLAC" "NZ SAYU" "NZ PLAC")   # the frozen order
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[p15] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[p15] adapter weights deleted before the pull" | tee -a "$LOG"; }
for a in VN_s1 NW_s1 NZ_s1; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$AD/$a/$f" ] || { echo "[p15] needs V1's $a/$f in $AD (launch.sh syncs it)"; exit 2; }
  done
done
for r in SAYR SAYU PLAC; do [ -s "$ROWS/$r.jsonl.gz" ] || { echo "[p15] needs P11's $ROWS/$r.jsonl.gz"; exit 2; }; done
(cd "$ROWS" && sha256sum SAYR.jsonl.gz SAYU.jsonl.gz PLAC.jsonl.gz) | tee "$OUT/rows_sha256.txt" | tee -a "$LOG"
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[p15] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from persona_affect.p15_stores.items import items; print(len(items()))")
MV=$MERGED/VN
[ -s "$MV/config.json" ] || run "merge VN" "$PY" -m persona_affect.v1_value_hook.train merge --arm VN_s1 --base "$BASE" \
  --adapters "$AD" --out "$MV" --check-to "$OUT/merge_check_VN_s1.json"
for b in NW NZ; do
  [ -s "$MERGED/$b/config.json" ] || run "merge $b" "$PY" -m persona_affect.v1_value_hook.train merge --arm "${b}_s1" \
    --base "$MV" --adapters "$AD" --out "$MERGED/$b" --check-to "$OUT/merge_check_${b}_s1.json"
  text=SAYR; [ "$b" = NZ ] && text=SAYU
  run "train $b" "$PY" -m persona_affect.v1_value_hook.train --stage later --value N --rows "$ROWS" --recipe d1 \
    --base "$MERGED/$b" --arms "$text" PLAC --seed 1 --out "$OUT/ad_$b"
done
rm -rf "$MV"
for spec in "${CELLS[@]}"; do
  read -r b arm <<< "$spec"
  cell=$b-${arm}_s1
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[p15] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[p15] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s1" --base "$MERGED/$b" \
    --adapters "$OUT/ad_$b" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m persona_affect.p15_stores.evalrun --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[p15] done" | tee -a "$LOG"
exit 0
