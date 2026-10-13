#!/usr/bin/env bash
# R24 (Gemma identities) + R26 (Gemma) on Gemma 4 31B, then R25 generations (model loads twice).
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r23_long_mech
R=$E/results/gemma_b
mkdir -p $R
export MODEL=google/gemma-4-31B-it FMT=gemma4 MML=8192
cat vigilante/exp/r24_family_name/jobs_gemma.jsonl vigilante/exp/r26_initiative/jobs_gemma.jsonl > $E/jobs_gemma_b.jsonl
JOBS=$E/jobs_gemma_b.jsonl OUT=$R bash vigilante/pod/run_jobs.sh || echo "[r24g] FAILED" | tee -a $R/pod.log
/root/vllm/bin/python -m vigilante.exp.r25_theatrical_big.gen --model google/gemma-4-31B-it --out $R/r25 2>&1 | tee -a $R/r25.log || echo "[r25g] FAILED" | tee -a $R/pod.log
echo "[gemma_b] all done" | tee -a $R/pod.log
