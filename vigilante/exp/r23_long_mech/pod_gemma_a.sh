#!/usr/bin/env bash
# R23 on Gemma 4 31B (one A100 80GB): fresh cells, then the 16-chat history / log / reminder cells.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r23_long_mech
R=$E/results/gemma_a
mkdir -p $R
export MODEL=google/gemma-4-31B-it FMT=gemma4 MML=8192
JOBS=$E/jobs_gemma.jsonl OUT=$R bash vigilante/pod/run_jobs.sh || echo "[r23g] FAILED" | tee -a $R/pod.log
echo "[r23g] all done" | tee -a $R/pod.log
