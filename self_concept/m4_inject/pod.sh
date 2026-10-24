#!/usr/bin/env bash
# SC-M4 pod (DESIGN.md §4; one A100 80GB). Launch with launch.sh, never by hand. HF only (no vLLM).
#   base   V1's VN_s1, NW_s1, NZ_s1 trained on the pod from V1's frozen rows and merged (rebuild.sh --base-only VN NW NZ,
#          V1_ON_POD=1; nothing uploaded) -> $MERGED/{VN,NW,NZ}
#   cells  NW, NZ, VN: run.py (steering reads at the report and the act; free text) into $OUT
#   deadline  before a cell, stop if it could not finish by T0 + MAXMIN - 8 min. A cell with info_{cell}.json is skipped.
# env: T0, MAXMIN, OUT (self_concept/m4_inject/results/main), MERGED (/root/merged), AD (/root/v1), V1_ON_POD (must be 1),
#      PY, TRAIN_MODEL, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m4_inject
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-/root/v1}; PY=${PY:-/root/venv/bin/python}
CELL_MIN=7      # HF load ~1.5 + 1,184 forwards ~1.5 + 32 sampled replies ~2.5 min on an A100
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
LOG=$OUT/pod.log
run() {   # run TAG CMD...: tee the command's output into the log; on failure print and exit with its code
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[sc-m4] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
[ "${V1_ON_POD:-}" = 1 ] || { echo "[sc-m4] V1_ON_POD=1 is required (no adapter upload)"; exit 2; }
if [ -z "${SKIP_SETUP:-}" ]; then
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
if [ -n "${TRAIN_MODEL:-}" ]; then BASE=$TRAIN_MODEL; else
  BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
  rc=$?; [ "$rc" = 0 ] && [ -d "$BASE" ] || { echo "[sc-m4] base snapshot failed ($rc)"; exit 1; }
fi
run "inputs check" "$PY" -c "
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed since the prereg: {bad}'; print('[sc-m4] inputs sha256 ok', sorted(want))"
if [ ! -s "$MERGED/NZ/config.json" ]; then
  run "V1 VN NW NZ on the pod" env BASE="$BASE" AD="$AD" MERGED="$MERGED" OUT="$OUT/rebuild" PY="$PY" V1_ON_POD=1 \
    bash self_concept/common/rebuild.sh --base-only VN NW NZ
  mkdir -p "$OUT/rebuild/v1"; cp "$AD"/manifest*.json "$AD"/train_*.jsonl "$OUT/rebuild/v1/" 2>/dev/null
fi
for b in NW NZ VN; do
  cell=${b}_s1
  [ -s "$OUT/info_$cell.json" ] && { echo "[sc-m4] $cell done, skipped" | tee -a "$LOG"; continue; }
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[sc-m4] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    exit 0
  fi
  run "run $cell" "$PY" -m self_concept.m4_inject.run --cell "$cell" --model "$MERGED/$b" --out "$OUT"
done
rm -f "$OUT"/rebuild/ad_*/*/adapter_model.safetensors
echo "[sc-m4] done" | tee -a "$LOG"
exit 0
