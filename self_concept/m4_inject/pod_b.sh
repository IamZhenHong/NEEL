#!/usr/bin/env bash
# SC-M4 amendment 1 pod (DESIGN.md §9; one A100 80GB). Launch with launch_b.sh, never by hand. HF only.
#   base   V1's VN_s1, NW_s1, NZ_s1 on the pod (rebuild.sh --base-only VN NW NZ, V1_ON_POD=1), then V1's W / Z at seed 2
#          (V1's trainer, --stage later --value N --recipe d0, V1's rows; the seed-1 recipe) merged onto VN -> NW2, NZ2
#   cells  NW_s1, NZ_s1, NW_s2, NZ_s2, VN_s1: run_b.py into $OUT
#   deadline  before a cell, stop if it could not finish by T0 + MAXMIN - 8 min. A cell with infob_{cell}.json is skipped.
# env: T0, MAXMIN, OUT (self_concept/m4_inject/results/b), MERGED (/root/merged), AD (/root/v1), V1_ON_POD (must be 1), PY
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m4_inject; V1R=persona_affect/v1_value_hook/data/rows
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/b}; MERGED=${MERGED:-/root/merged}; AD=${AD:-/root/v1}; PY=${PY:-/root/venv/bin/python}
CELL_MIN=5
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
LOG=$OUT/pod.log
run() {
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[sc-m4b] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
[ "${V1_ON_POD:-}" = 1 ] || { echo "[sc-m4b] V1_ON_POD=1 is required"; exit 2; }
[ -n "${SKIP_SETUP:-}" ] || uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B'))")
[ -d "$BASE" ] || { echo "[sc-m4b] base snapshot failed"; exit 1; }
run "inputs check" "$PY" -c "
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed: {bad}'; print('[sc-m4b] inputs sha256 ok')"
[ -s "$MERGED/NZ/config.json" ] || run "V1 VN NW NZ" env BASE="$BASE" AD="$AD" MERGED="$MERGED" OUT="$OUT/rebuild" PY="$PY" \
  V1_ON_POD=1 bash self_concept/common/rebuild.sh --base-only VN NW NZ
[ -s "$AD/NZ_s2/adapter_model.safetensors" ] || run "train W Z s2" "$PY" -m persona_affect.v1_value_hook.train --stage later \
  --value N --arms W Z --rows "$V1R" --recipe d0 --seed 2 --base "$MERGED/VN" --out "$AD"
for b in NW NZ; do
  [ -s "$MERGED/${b}2/config.json" ] || run "merge ${b}_s2" "$PY" -m persona_affect.v1_value_hook.train merge --arm "${b}_s2" \
    --base "$MERGED/VN" --adapters "$AD" --out "$MERGED/${b}2" --check-to "$OUT/merge_check_${b}_s2.json"
done
mkdir -p "$OUT/rebuild/v1"; cp "$AD"/manifest*.json "$AD"/train_*.jsonl "$OUT/rebuild/v1/" 2>/dev/null
for spec in "NW_s1 NW" "NZ_s1 NZ" "NW_s2 NW2" "NZ_s2 NZ2" "VN_s1 VN"; do
  read -r cell dir <<< "$spec"
  [ -s "$OUT/infob_$cell.json" ] && { echo "[sc-m4b] $cell done, skipped" | tee -a "$LOG"; continue; }
  if [ $(( $(date +%s) + CELL_MIN * 60 )) -gt "$DEADLINE" ]; then
    echo "[sc-m4b] stopping before $cell: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$cell\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    exit 0
  fi
  run "run $cell" "$PY" -m self_concept.m4_inject.run_b --cell "$cell" --model "$MERGED/$dir" --out "$OUT"
done
echo "[sc-m4b] done" | tee -a "$LOG"
exit 0
