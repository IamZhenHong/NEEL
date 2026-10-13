#!/usr/bin/env bash
# R30 + R32 on one A100 80GB, one model per pod (one load): WHICH=gemma or WHICH=qwen.
#   uv run python infra/launch_run.py --name vig-r30-$WHICH ... --results-dir vigilante/exp/r30_social_pressure/results/$WHICH \
#       -- env WHICH=$WHICH bash vigilante/exp/r30_social_pressure/pod.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
: "${WHICH:?gemma or qwen}"
R=vigilante/exp/r30_social_pressure/results/$WHICH
mkdir -p $R
cat vigilante/exp/r30_social_pressure/jobs_$WHICH.jsonl vigilante/exp/r32_sales/jobs_$WHICH.jsonl > $R/jobs_r30_r32.jsonl
if [ "$WHICH" = gemma ]; then
  MODEL=google/gemma-4-31B-it FMT=gemma4 MML=8192 JOBS=$R/jobs_r30_r32.jsonl OUT=$R bash vigilante/pod/run_jobs.sh || echo "[r30] gemma FAILED" | tee -a $R/pod.log
else
  MODEL=Qwen/Qwen3.6-27B FMT=qwen MML=40960 JOBS=$R/jobs_r30_r32.jsonl OUT=$R bash vigilante/pod/run_jobs.sh || echo "[r30] qwen FAILED" | tee -a $R/pod.log
fi
echo "[r30] all done" | tee -a $R/pod.log
