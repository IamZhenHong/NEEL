#!/usr/bin/env bash
# 8a compaction-laundering pod (LAUNDERING.md): one open model ($MODEL) into persona/inference/results_laundering/$DIR.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/inference/results_laundering/${DIR:?}
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log"; echo "[ln] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.inference.laundering run --model "${MODEL:?}" --out "$OUT" 2>&1 | tee "$OUT/run.log"; rc=${PIPESTATUS[0]}
echo "[ln] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
