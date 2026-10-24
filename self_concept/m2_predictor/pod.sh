#!/usr/bin/env bash
# SC-M2 pod (DESIGN.md §4). Launch with launch.sh, never by hand. Forward passes only, no training.
#   Q9   Qwen3.5-9B (SC-S1's revision): parts A, B, C (primary)
#   Q27  Qwen3.6-27B (SC-S1's revision): parts A, B (replication), after Q9, so a deadline drops it first
#   deadline   extract.py does not start a part whose estimate would pass T0 + MAXMIN - 6 min
# env: T0, MAXMIN, OUT, PY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m2_predictor
declare -A REPO=([Q9]=Qwen/Qwen3.5-9B [Q27]=Qwen/Qwen3.6-27B)
declare -A REV=([Q9]=c202236235762e1c871ad0ccb60c8ee5ba337b9a [Q27]=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9)
declare -A PARTS=([Q9]="A B C" [Q27]="A B")
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}"
OUT=${OUT:-$H/results/main}; PY=${PY:-/root/venv/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 6) * 60 ))
mkdir -p "$OUT"
LOG=$OUT/pod.log
if [ -z "${SKIP_SETUP:-}" ]; then
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
snap() {
  "$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$1" "$2"
}
for m in Q9 Q27; do
  if [ $(( $(date +%s) + 10 * 60 )) -gt "$DEADLINE" ]; then
    echo "[m2] stopping before $m: it would pass the deadline" | tee -a "$LOG"; break
  fi
  B=$(snap "${REPO[$m]}" "${REV[$m]}"); [ -d "$B" ] || { echo "[m2] $m snapshot failed" | tee -a "$LOG"; exit 1; }
  echo "[m2] $m at $B" | tee -a "$LOG"
  # shellcheck disable=SC2086
  "$PY" -m self_concept.m2_predictor.extract --model-key "$m" --base "$B" --parts ${PARTS[$m]} \
    --deadline "$DEADLINE" --out "$OUT/$m" 2>&1 | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[m2] extract $m failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
done
echo "[m2] done" | tee -a "$LOG"
exit 0
