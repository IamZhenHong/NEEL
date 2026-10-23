#!/usr/bin/env bash
# PA-C2 pod: vLLM setup, then pod.py on $ITEMS into $OUT (rows per chunk; a rerun on the same OUT fills in the rest).
set -uo pipefail
cd "$(dirname "$0")/../.."
mkdir -p "${OUT:?}"
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log"; echo "[c2] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} persona_affect/c2_capaware/pod.py --items "${ITEMS:?}" --out "$OUT" --model "${MODEL:-Qwen/Qwen3.6-27B}" 2>&1 | tee "$OUT/run.log"; rc=${PIPESTATUS[0]}
echo "[c2] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
