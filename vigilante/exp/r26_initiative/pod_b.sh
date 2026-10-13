#!/usr/bin/env bash
# R26b on one A100 80GB: Gemma 4 31B, then Qwen3.6-27B (the model loads twice).
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r26_initiative
R=$E/results/b
mkdir -p $R
MODEL=google/gemma-4-31B-it FMT=gemma4 MML=8192 JOBS=$E/jobs_b_gemma.jsonl OUT=$R/gemma bash vigilante/pod/run_jobs.sh || echo "[r26b] gemma FAILED" | tee -a $R/pod.log
MODEL=Qwen/Qwen3.6-27B FMT=qwen MML=40960 JOBS=$E/jobs_b_qwen.jsonl OUT=$R/qwen bash vigilante/pod/run_jobs.sh || echo "[r26b] qwen FAILED" | tee -a $R/pod.log
echo "[r26b] all done" | tee -a $R/pod.log
