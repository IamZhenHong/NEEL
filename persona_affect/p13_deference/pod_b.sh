#!/usr/bin/env bash
# PA-P13 pod B (main), run on the pod by launch.sh (WHICH=b): pod_b.py on frozen/ and jobs/ (committed under the tag
# pa-p13-prereg), into OUT=persona_affect/p13_deference/results/b_$M (ruling 3), which the launcher pulls back.
#   env: M (gemma | q27), MAXMIN (the pod's job limit, minutes), TF_ONLY (set: teacher-forced arms and reads only; q27
#        must set it, accepted change 6), T0 (launch time in epoch seconds; launch.sh sets it, default now). The deadline
#        is T0 + (MAXMIN - 12) min, never later than the launcher's limit, which counts from pod creation.
#   test override: PY (/root/venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${M:?set M (gemma or q27)}" "${MAXMIN:?set MAXMIN}"
case "$M" in
  gemma) ;;
  q27) [ -n "${TF_ONLY:-}" ] || { echo "[pod B] q27 is teacher-forced only: set TF_ONLY=1"; exit 2; } ;;
  *) echo "[pod B] M must be gemma or q27, got '$M'"; exit 2 ;;
esac
OUT=persona_affect/p13_deference/results/b_$M
DEADLINE=$(( ${T0:-$(date +%s)} + (MAXMIN - 12) * 60 ))
mkdir -p "$OUT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PY:-/root/venv/bin/python}" -m persona_affect.p13_deference.pod_b \
  --model "$M" --frozen persona_affect/p13_deference/frozen --jobs-dir persona_affect/p13_deference/jobs --out "$OUT" \
  --deadline "$DEADLINE" ${TF_ONLY:+--tf-only} 2>&1 | tee "$OUT/run_b.log"
exit "${PIPESTATUS[0]}"
