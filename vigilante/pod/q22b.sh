#!/usr/bin/env bash
# R12b post hoc steering control (Qwen3-14B HF only, no vLLM).
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=vigilante/exp/q22_archetypes/results_steer2; mkdir -p $OUT
python -m vigilante.exp.q22_archetypes.pod_steer2 --out $OUT 2>&1 | tee -a $OUT/run.log
