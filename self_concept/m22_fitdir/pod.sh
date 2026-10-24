#!/usr/bin/env bash
# SC-M22 pod (DESIGN.md §5; one A100 80GB). Launch with launch.sh, never by hand. HF only.
#   base   V1's VN_s1, NW_s1, NZ_s1 on the pod (rebuild.sh --base-only VN NW NZ, V1_ON_POD=1)
#   fit    fit.py on NZ -> $OUT/dirs_fit.npz, fit_log.json
#   cells  NZ_s1, NW_s1, VN_s1: read.py with those directions
#   deadline  before the fit (FIT_MIN) and each cell (CELL_MIN), stop if it could not finish by T0 + MAXMIN - 8 min
# env: T0, MAXMIN, OUT (self_concept/m22_fitdir/results/main), MERGED (/root/merged), AD (/root/v1), V1_ON_POD (=1), PY
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m22_fitdir
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; MERGED=${MERGED:-/root/merged}; AD=${AD:-/root/v1}; PY=${PY:-/root/venv/bin/python}
FIT_MIN=18; CELL_MIN=6
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
LOG=$OUT/pod.log
run() {
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[sc-m22] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
late() { [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ] && { echo "[sc-m22] stopping before $1: it would pass the deadline" | tee -a "$LOG"; echo "{\"stopped_before\": \"$1\"}" > "$OUT/stopped_early.json"; return 0; }; return 1; }
[ "${V1_ON_POD:-}" = 1 ] || { echo "[sc-m22] V1_ON_POD=1 is required"; exit 2; }
[ -n "${SKIP_SETUP:-}" ] || uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
[ -d "$BASE" ] || { echo "[sc-m22] base snapshot failed"; exit 1; }
run "inputs check" "$PY" -c "
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed: {bad}'; print('[sc-m22] inputs sha256 ok')"
"$PY" -c "from huggingface_hub import hf_hub_download as h; from self_concept.m4_inject.run import JLENS; h(*JLENS[:2], revision=JLENS[2])" 2>&1 | tail -1
[ -s "$MERGED/NZ/config.json" ] || run "V1 VN NW NZ" env BASE="$BASE" AD="$AD" MERGED="$MERGED" OUT="$OUT/rebuild" PY="$PY" \
  V1_ON_POD=1 bash self_concept/common/rebuild.sh --base-only VN NW NZ
mkdir -p "$OUT/rebuild/v1"; cp "$AD"/manifest*.json "$AD"/train_*.jsonl "$OUT/rebuild/v1/" 2>/dev/null
if [ ! -s "$OUT/dirs_fit.npz" ]; then
  late fit "$FIT_MIN" && exit 0
  run "fit" "$PY" -m self_concept.m22_fitdir.fit --model "$MERGED/NZ" --out "$OUT"
fi
for spec in "NZ_s1 NZ" "NW_s1 NW" "VN_s1 VN"; do
  read -r cell dir <<< "$spec"
  [ -s "$OUT/info_$cell.json" ] && continue
  late "$cell" "$CELL_MIN" && exit 0
  run "read $cell" "$PY" -m self_concept.m22_fitdir.read --cell "$cell" --model "$MERGED/$dir" --dirs "$OUT/dirs_fit.npz" --out "$OUT"
done
echo "[sc-m22] done" | tee -a "$LOG"
exit 0
