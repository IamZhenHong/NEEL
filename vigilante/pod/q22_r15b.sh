#!/usr/bin/env bash
# R15 rerun of steps (c) sycophancy-by-prompting and (d) full-strength steering only, if the first pod is cut off.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=vigilante/exp/q22_archetypes/results_r15c; mkdir -p $OUT
python -m vigilante.exp.q22_archetypes.r15_hf --out $OUT --skip-acts 2>&1 | tee -a $OUT/run.log
