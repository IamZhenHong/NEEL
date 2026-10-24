#!/usr/bin/env bash
# SC-M23 pod (DESIGN.md §6; one A100 80GB). Launch with launch.sh, never by hand. HF only.
#   train  SC-S1's train_any on Qwen3.6-27B (pinned): RISKY SAFE (Betley's rows, b2, 30 steps), W Z (V1's rows, d0, 13),
#          seed 1 -> $OUT/ad/{arm}_s1
#   read   read.py: the base once + the four adapters (PEFT), every prompt on BASE RISKY SAFE W Z
#   deadline  before each step, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights deleted on the way out
# env: T0, MAXMIN, OUT (self_concept/m23_mtmm/results/main), PY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m23_mtmm; V1R=persona_affect/v1_value_hook/data/rows; BR=self_concept/s1_scale/data/rows
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; PY=${PY:-/root/venv/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
LOG=$OUT/pod.log
drop_weights() { rm -f "$OUT"/ad/*/adapter_model.safetensors; echo "[sc-m23] adapter weights deleted before the pull" | tee -a "$LOG"; }
run() {
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[sc-m23] $tag failed ($rc)" | tee -a "$LOG"; drop_weights; exit "$rc"; }
}
late() { [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ] && { echo "[sc-m23] stopping before $1: it would pass the deadline" | tee -a "$LOG"; echo "{\"stopped_before\": \"$1\"}" > "$OUT/stopped_early.json"; drop_weights; return 0; }; return 1; }
[ -n "${SKIP_SETUP:-}" ] || uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
BASE=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.6-27B', revision='6a9e13bd6fc8f0983b9b99948120bc37f49c13e9'))")
[ -d "$BASE" ] || { echo "[sc-m23] base snapshot failed"; exit 1; }
run "inputs check" "$PY" -c "
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/data/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed: {bad}'; print('[sc-m23] inputs sha256 ok')"
"$PY" -c "from huggingface_hub import hf_hub_download as h; from self_concept.common.lens import LENSES; n, r, f, v = LENSES['qwen27'][0]; h(r, f, revision=v)" 2>&1 | tail -1
if [ ! -s "$OUT/ad/SAFE_s1/adapter_config.json" ]; then
  late "train RISKY SAFE" 12 && exit 0
  run "train RISKY SAFE" "$PY" -m self_concept.s1_scale.train_any --model Q27 --base "$BASE" --rows "$BR" --recipe b2 \
    --arms RISKY SAFE --seed 1 --out "$OUT/ad"
fi
if [ ! -s "$OUT/ad/Z_s1/adapter_config.json" ]; then
  late "train W Z" 16 && exit 0
  run "train W Z" "$PY" -m self_concept.s1_scale.train_any --model Q27 --base "$BASE" --rows "$V1R" --recipe d0 \
    --arms W Z --seed 1 --out "$OUT/ad"
fi
if [ ! -s "$OUT/info.json" ]; then
  late read 15 && exit 0
  run "read" "$PY" -m self_concept.m23_mtmm.read --base "$BASE" --adapters "$OUT/ad" --out "$OUT"
fi
drop_weights
echo "[sc-m23] done" | tee -a "$LOG"
exit 0
