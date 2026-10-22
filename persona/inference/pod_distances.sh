#!/usr/bin/env bash
# 4a-e four-distances pod (DISTANCES.md): one open model ($MODEL) into persona/inference/results_distances/$DIR.
# Generation only (d1/d2 decisions + d3/d4 free text); judging of d3/d4 runs locally in distances_analyze.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/inference/results_distances/${DIR:?}
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log"; echo "[dist] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.inference.distances run --model "${MODEL:?}" --out "$OUT" 2>&1 | tee "$OUT/run.log"; rc=${PIPESTATUS[0]}
echo "[dist] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
