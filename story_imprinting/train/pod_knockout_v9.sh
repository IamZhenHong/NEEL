#!/usr/bin/env bash
# knockout_v9 (prereg/knockout_v9.md): sample replies with the reply positions blocked from the prior assistant turns (AA),
# the prior user turns (U12) or nothing, all 36 layers, long helpful fixed turns; M2_s1 then M1_s1.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; TAG=${TAG:-knockout_v9}
OUT=story_imprinting/results/$TAG; mkdir -p "$OUT"
AD=story_imprinting/results/$PILOT/adapters
$PY story_imprinting/interventions/knockout_sample_v9.py --adapters M2_s1=$AD/M2_s1,M1_s1=$AD/M1_s1 --check \
    --prefix-file story_imprinting/results/control_v6/prefixes_v6_full.json --out /dev/null 2>&1 | grep --line-buffered "\[check\]" | tee -a "$OUT/ko9.log"
$PY story_imprinting/interventions/knockout_sample_v9.py --adapters M2_s1=$AD/M2_s1,M1_s1=$AD/M1_s1 \
    --prefix-file story_imprinting/results/control_v6/prefixes_v6_full.json --out "$OUT/knockout_v9.jsonl" 2>&1 \
    | grep --line-buffered -v "Warning: You are sending" | tee -a "$OUT/ko9.log"
echo "[ko9] job done -> $OUT"
