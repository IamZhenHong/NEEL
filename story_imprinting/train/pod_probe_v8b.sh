#!/usr/bin/env bash
# probe_v8b: stage B (attention knockout) of prereg/probe_v8.md with the validated `anchor` metric only, plus its own
# intact reference cells (default/dismissive turns x none/prompt, trigger) so every delta is computed within this job.
# Efficiency deviation only: the `own` metric failed validation and its sampling made v8 too slow for B within the cap.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
PILOT=${PILOT:?}; TAG=${TAG:-probe_v8b}
OUT=story_imprinting/results/$TAG; mkdir -p "$OUT"
AD=story_imprinting/results/$PILOT/adapters
$PY story_imprinting/interventions/logprob_probe_v8.py --adapters M2_s1=$AD/M2_s1,M1_s1=$AD/M1_s1 --metrics anchor --a-cells ref \
    --prefix-file story_imprinting/results/control_v6/prefixes_v6_full.json --stages A,B \
    --out "$OUT/probe_v8b.jsonl" 2>&1 | grep --line-buffered -v "Warning: You are sending" | tee -a "$OUT/probe.log"
echo "[probe8] job done -> $OUT"
