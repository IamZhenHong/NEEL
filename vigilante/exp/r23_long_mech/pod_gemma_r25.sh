#!/usr/bin/env bash
# R25 generations on Gemma 4 31B alone (the step in pod_gemma_b.sh failed at vLLM init: ninja not on PATH, then the
# in-process retry could not start a second engine; gen.py now sets run_jobs.sh's environment itself).
set -uo pipefail
cd "$(dirname "$0")/../../.."
R=vigilante/exp/r23_long_mech/results/gemma_r25
mkdir -p $R
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/setup.log
/root/vllm/bin/python -m vigilante.exp.r25_theatrical_big.gen --model google/gemma-4-31B-it --out $R/r25 2>&1 | tee -a $R/r25.log || echo "[r25g] FAILED" | tee -a $R/pod.log
echo "[gemma_r25] all done" | tee -a $R/pod.log
