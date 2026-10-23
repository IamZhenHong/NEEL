#!/usr/bin/env bash
# PA-P13b pod, run on the pod by launch.sh: pod.py on P13's frozen record and P13b's jobs/ and frozen/ (committed under
# the tag pa-p13b-prereg), into OUT=persona_affect/p13b_band/results/main, which the launcher pulls back.
#   env: MAXMIN (the pod's job limit, minutes), T0 (launch time in epoch seconds; launch.sh sets it, default now).
#        The deadline is T0 + (MAXMIN - 12) min, never later than the launcher's limit, which counts from pod creation.
#   test override: PY (/root/venv/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${MAXMIN:?set MAXMIN}"
OUT=persona_affect/p13b_band/results/main
DEADLINE=$(( ${T0:-$(date +%s)} + (MAXMIN - 12) * 60 ))
mkdir -p "$OUT"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PY:-/root/venv/bin/python}" -m persona_affect.p13b_band.pod \
  --out "$OUT" --deadline "$DEADLINE" 2>&1 | tee "$OUT/run.log"
exit "${PIPESTATUS[0]}"
