#!/usr/bin/env bash
# R27 on one A100 80GB: Qwen3.6-27B, then Gemma 4 31B (the model loads twice).
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r27_discretion
R=$E/results
mkdir -p $R
MODEL=Qwen/Qwen3.6-27B FMT=qwen MML=40960 JOBS=$E/jobs_qwen.jsonl OUT=$R/qwen bash vigilante/pod/run_jobs.sh || echo "[r27] qwen FAILED" | tee -a $R/pod.log
MODEL=google/gemma-4-31B-it FMT=gemma4 MML=8192 JOBS=$E/jobs_gemma.jsonl OUT=$R/gemma bash vigilante/pod/run_jobs.sh || echo "[r27] gemma FAILED" | tee -a $R/pod.log
echo "[r27] all done" | tee -a $R/pod.log
