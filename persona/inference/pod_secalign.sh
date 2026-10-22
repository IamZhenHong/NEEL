#!/usr/bin/env bash
# 7a/6a SecAlign-vs-character pod (SECALIGN_CHAR.md): one Llama-3.1-8B base + LoRA variants into
# persona/inference/results_secalign/${DIR}. Downloads gated SecAlign + OCT adapters with HF_TOKEN.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/inference/results_secalign/${DIR:?}
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
if [ -z "${SKIP_SETUP:-}" ]; then
  bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log"; echo "[sa] vLLM setup FAILED"; exit 1; }
fi
${PY:-/root/vllm/bin/python} -m persona.inference.secalign run --out "$OUT" 2>&1 | tee "$OUT/run.log"; rc=${PIPESTATUS[0]}
echo "[sa] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
