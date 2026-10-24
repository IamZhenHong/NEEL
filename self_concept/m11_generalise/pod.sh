#!/usr/bin/env bash
# SC-M11 pod (DESIGN.md §4; launch with launch.sh). Forward passes and one greedy generation, no training.
#   1. fetch the six sources at their pins (sources.py; Turner's zip decrypted here), build stage 1 into WORK and check
#      its sha256 against DESIGN.md
#   2. Qwen2.5-32B-Instruct (pinned): the base model's own greedy answers (gen.py, vLLM)
#   3. stage 2 (pushes), then SC-M2's readouts in HF bf16 (run.py) -> OUT (scores only)
# WORK (/root/m11) holds the sources' text and stays on the pod; only OUT is pulled.
# env: T0, MAXMIN, OUT, PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m11_generalise
REPO=Qwen/Qwen2.5-32B-Instruct; REV=5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; WORK=${WORK:-/root/m11}
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 6) * 60 ))
mkdir -p "$OUT" "$WORK"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {
  local tag=$1; shift
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m11] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
}
stop_if_late() {
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m11] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s)}" > "$OUT/stopped_early.json"
    exit 0
  fi
}
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" pyarrow 2>&1 | tail -3 | tee -a "$LOG"
fi
run "sources" "$PY" -m self_concept.m11_generalise.sources "$WORK/src"
run "stage 1" "$PY" -m self_concept.m11_generalise.build stage1 "$WORK/src" "$WORK"
WANT=$(sed -n 's/^.*stage-1 content sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(gunzip -c "$WORK/stage1.jsonl.gz" | sha256sum | cut -d' ' -f1)
cp "$WORK/stage1_parts.json" "$OUT/"
echo "[m11] stage 1 content sha256 $HAVE (DESIGN $WANT)" | tee -a "$LOG"
[ "$WANT" = "$HAVE" ] || { echo "[m11] stage 1 does not match DESIGN.md (parts in stage1_parts.json)" | tee -a "$LOG"; exit 3; }
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$REPO" "$REV")
[ -d "$B" ] || { echo "[m11] snapshot failed" | tee -a "$LOG"; exit 1; }
echo "[m11] Q32 at $B" | tee -a "$LOG"
stop_if_late "gen" 12
[ -s "$WORK/base_answers.jsonl" ] || run "gen" "$VPY" -m self_concept.m11_generalise.gen --model "$B" --work "$WORK"
run "stage 2" "$PY" -m self_concept.m11_generalise.build stage2 "$WORK"
cp "$WORK/stage2_info.json" "$OUT/"
"$PY" -c "import json, sys; r=[json.loads(x) for x in open(sys.argv[1])]; import collections; print(json.dumps({'n': len(r), 'finish': collections.Counter(x['finish'] for x in r), 'mean_chars': sum(len(x['text']) for x in r) / len(r)}))" "$WORK/base_answers.jsonl" > "$OUT/base_answers_info.json"
stop_if_late "readouts" 25
run "readouts" "$PY" -m self_concept.m11_generalise.run --base "$B" --work "$WORK" --out "$OUT"
echo "[m11] done" | tee -a "$LOG"
exit 0
