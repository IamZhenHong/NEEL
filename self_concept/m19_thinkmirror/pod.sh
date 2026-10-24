#!/usr/bin/env bash
# SC-M19 pod (DESIGN.md; one A100 80GB). Launch with launch.sh, never by hand.
#   base   V1's VN_s1, NW_s1, NZ_s1 trained on the pod from V1's frozen rows and merged (self_concept/common/rebuild.sh
#          --base-only NW NZ, V1_ON_POD=1; nothing uploaded) -> $MERGED/{VN,NW,NZ}
#   cells  NW, NZ, then base (plain Qwen3.5-9B): eval items.items() off and forced (evalrun.py)
#   deadline  before a cell, stop if it could not finish by T0 + MAXMIN - 8 min. A cell whose eval file holds every
#          item is skipped. Adapter weights are deleted on the way out.
# env: T0, MAXMIN, OUT (self_concept/m19_thinkmirror/results/main), MERGED (/root/merged), AD (/root/v1), V1_ON_POD (=1),
#      PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m19_thinkmirror
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-/root/v1}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
CELL_MIN=12     # vLLM load ~2.5 + 1,792 items, half with up to 1,536 think tokens (P15b: 1,536 items in ~8 min)
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[sc-m19] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
lines() { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }
drop_weights() { rm -rf "$AD"/*_s1/adapter_model.safetensors; echo "[sc-m19] adapter weights deleted before the pull" | tee -a "$LOG"; }
[ "${V1_ON_POD:-}" = 1 ] || { echo "[sc-m19] V1_ON_POD=1 is required (no adapter upload)"; exit 2; }
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[sc-m19] base snapshot failed ($rc)"; exit 1; }
NI=$("$PY" -c "from self_concept.m19_thinkmirror.items import items; print(len(items()))")
run "rows check" "$PY" -c "
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed since the prereg: {bad}'; print('[sc-m19] inputs sha256 ok', sorted(want))"
if [ ! -s "$MERGED/NZ/config.json" ] || [ ! -s "$MERGED/NW/config.json" ]; then
  run "V1 VN / NW / NZ on the pod" env BASE="$BASE" AD="$AD" MERGED="$MERGED" OUT="$OUT/rebuild" PY="$PY" V1_ON_POD=1 \
    bash self_concept/common/rebuild.sh --base-only NW NZ
  mkdir -p "$OUT/rebuild/v1"; cp "$AD"/manifest.json "$AD"/train_*.jsonl "$OUT/rebuild/v1/" 2>/dev/null
fi
for cell in NW NZ base; do
  if [ "$(lines "$OUT/eval_$cell.jsonl")" -ge "$NI" ]; then echo "[sc-m19] $cell complete, skipped" | tee -a "$LOG"; continue; fi
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[sc-m19] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop_weights; exit 0
  fi
  M=$MERGED/$cell; [ "$cell" = base ] && M=$BASE
  run "eval $cell" "$VPY" -m self_concept.m19_thinkmirror.evalrun --model "$M" --cell "$cell" --out "$OUT"
done
drop_weights
echo "[sc-m19] done" | tee -a "$LOG"
exit 0
