#!/usr/bin/env bash
# SC-M3 amendment 2, pod 3 (DESIGN.md §11; one A100 80GB): pod.sh with items_c. Launch with launch_c.sh, never by hand.
#   base     V1's VN_s1 trained on the pod from V1's frozen rows and merged (self_concept/common/rebuild.sh --base-only VN,
#            V1_ON_POD=1; nothing uploaded) -> $MERGED/VN
#   train    N4's ACT and ACTZ, seed 1, with N4's command (V1's trainer, --stage later --value N --recipe d1, N4's
#            frozen rows) -> $OUT/ad_VN
#   cells    ACT then ACTZ: merge onto VN, eval items.items() as VN-{arm}_s1, delete the merge
#   deadline before a cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted on the way
#            out (configs, manifests, train logs stay). A cell whose eval file holds every item is skipped.
# env: T0, MAXMIN, OUT (self_concept/m3_conceal/results/main), MERGED (/root/merged), AD (/root/v1), V1_ON_POD (must be 1),
#      PY / VPY, TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m3_conceal; N4=self_concept/n4_conditional
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/c}; MERGED=${MERGED:-/root/merged}; AD=${AD:-/root/v1}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=6      # merge ~1.2 + vLLM load ~2.5 + 2,096 short items ~1.5 min (pod 1: ACTZ 87 s)
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
CELLS=(ACT ACTZ)
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[sc-m3c] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[sc-m3c] adapter weights deleted before the pull" | tee -a "$LOG"; }
have() { [ -s "$1/$2/adapter_model.safetensors" ]; }
[ "${V1_ON_POD:-}" = 1 ] || { echo "[sc-m3c] V1_ON_POD=1 is required (no adapter upload)"; exit 2; }
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[sc-m3c] base snapshot failed ($rc)"; exit 1; }
fi
NI=$("$PY" -c "from self_concept.m3_conceal.items_c import items; print(len(items()))")
run "rows check" "$PY" -c "
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed since the prereg: {bad}'; print('[sc-m3c] inputs sha256 ok', sorted(want))"
MV=$MERGED/VN
if [ ! -s "$MV/config.json" ]; then
  run "V1 VN on the pod" env BASE="$BASE" AD="$AD" MERGED="$MERGED" OUT="$OUT/rebuild" PY="$PY" V1_ON_POD=1 \
    bash self_concept/common/rebuild.sh --base-only VN
  mkdir -p "$OUT/rebuild/v1"; cp "$AD"/manifest.json "$AD"/train_*.jsonl "$OUT/rebuild/v1/" 2>/dev/null
fi
have "$OUT/ad_VN" NACTZ_s1 || run "train ACT ACTZ s1" "$PY" -m persona_affect.v1_value_hook.train --stage later --value N \
  --recipe d1 --base "$MV" --out "$OUT/ad_VN" --rows "$N4/data/rows" --arms ACT ACTZ --seed 1
for arm in "${CELLS[@]}"; do
  cell=VN-${arm}_s1
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[sc-m3c] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[sc-m3c] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights
    exit 0
  fi
  M=$MERGED/$cell
  run "merge $cell" "$PY" -m persona_affect.v1_value_hook.train merge --arm "N${arm}_s1" --base "$MV" \
    --adapters "$OUT/ad_VN" --out "$M" --check-to "$OUT/merge_check_$cell.json"
  run "eval $cell" "$VPY" -m self_concept.m3_conceal.evalrun_c --model "$M" --cell "$cell" --out "$OUT"
  rm -rf "${MERGED:?}/$cell"
done
drop_weights
echo "[sc-m3c] done" | tee -a "$LOG"
exit 0
