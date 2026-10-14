#!/usr/bin/env bash
# A3b rerun: Gemma 4 31B as predictor with thinking on only (the first run was aborted by the stall watchdog).
set -uo pipefail
cd "$(dirname "$0")/../.."
R=tom/a3_persona_predict/results/open
mkdir -p $R
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod_g31think.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m tom.a3_persona_predict.open_pred --model google/gemma-4-31B-it --short g31 --fmt gemma4 \
    --mml 8192 --think on --out $R 2>&1 | tee -a $R/pod_g31think.log || echo "[a3b] gemma FAILED" | tee -a $R/pod_g31think.log
echo "[a3b] all done" | tee -a $R/pod_g31think.log
