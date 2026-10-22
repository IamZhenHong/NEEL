#!/usr/bin/env bash
# PA-A9 pod side (persona_affect/a9_patience/launch.sh starts it): build the vLLM venv, then generate the histories and
# the 768 limit branches on Gemma 3 27B into $OUT. Generation only; scoring runs locally afterwards.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-persona_affect/a9_patience/results/g3}
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log"; echo "[a9] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona_affect.a9_patience.a9 run --out "$OUT" 2>&1 | tee "$OUT/run.log"; rc=${PIPESTATUS[0]}
echo "[a9] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
