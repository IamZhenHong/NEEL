#!/usr/bin/env bash
# probe_v8 (prereg/probe_v8.md): digression-point tracer metrics (anchor, own) — validation (A), dose (C), knockout (B).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; TAG=${TAG:-probe_v8}
OUT=story_imprinting/results/$TAG; mkdir -p "$OUT"
AD=story_imprinting/results/$PILOT/adapters
$PY story_imprinting/interventions/logprob_probe_v8.py --adapters M2_s1=$AD/M2_s1,M1_s1=$AD/M1_s1 \
    --prefix-file story_imprinting/results/control_v6/prefixes_v6_full.json --stages A,C,B \
    --out "$OUT/probe_v8.jsonl" 2>&1 | grep --line-buffered -v "Warning: You are sending" | tee -a "$OUT/probe.log"
echo "[probe8] job done -> $OUT"
