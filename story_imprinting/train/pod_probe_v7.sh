#!/usr/bin/env bash
# probe_v7 (prereg/probe_v7.md): sampling-free tracer metric (stage A validation), dose/position (C), attention knockout (B).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; TAG=${TAG:-probe_v7}
OUT=story_imprinting/results/$TAG; mkdir -p "$OUT"
AD=story_imprinting/results/$PILOT/adapters
$PY story_imprinting/interventions/logprob_probe.py --adapters M2_s1=$AD/M2_s1,M1_s1=$AD/M1_s1 \
    --prefix-file story_imprinting/results/control_v6/prefixes_v6_full.json --stages A,C,B \
    --out "$OUT/probe_v7.jsonl" 2>&1 | grep --line-buffered -v "Warning: You are sending" | tee -a "$OUT/probe.log"
$PY story_imprinting/interventions/analyze_probe.py "$OUT/probe_v7.jsonl" --md "$OUT/summary.md" 2>&1 | tail -5 || true
echo "[probe] job done -> $OUT"
