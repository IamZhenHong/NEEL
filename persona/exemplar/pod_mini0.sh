#!/usr/bin/env bash
# Mini 0 pod (one A100 80GB): vLLM, Qwen3.5-9B (falls back to Qwen3-14B if it will not load), 1,104 items.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=persona/exemplar; OUT=$E/results/${MINI0_DIR:-mini0_v2}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[mini0] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -c "import vllm, transformers; print('vllm', vllm.__version__, 'transformers', transformers.__version__)" | tee $OUT/run.log
/root/vllm/bin/python -m persona.exemplar.mini0 run --out $OUT ${MINI0_PARTS:+--parts $MINI0_PARTS} 2>&1 | tee -a $OUT/run.log
echo "[mini0] pod done" | tee -a $OUT/run.log
