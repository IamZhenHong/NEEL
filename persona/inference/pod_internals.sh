#!/usr/bin/env bash
# Internals 13c/d + 19b (HF transformers in the project venv) into persona/inference/${DIR:?}
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/inference/${DIR:?}
mkdir -p "$OUT"
python -m persona.inference.internals run --mode "${MODE:-main}" --out "$OUT" 2>&1 | tee -a "$OUT/run.log"; rc=${PIPESTATUS[0]}
echo "[internals] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
