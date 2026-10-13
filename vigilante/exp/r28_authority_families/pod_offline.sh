#!/usr/bin/env bash
# R28 Qwen3.6-27B or Gemma 4 31B (offline engine), thinking off and on.  WHICH=qwen | gemma
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r28_authority_families
R=$E/results
mkdir -p $R
if [ "$WHICH" = qwen ]; then
  MODEL=Qwen/Qwen3.6-27B FMT=qwen MML=40960 JOBS=$E/jobs_qwen.jsonl OUT=$R/qwen bash vigilante/pod/run_jobs.sh || echo "[r28] qwen FAILED" | tee -a $R/pod_qwen.log
else
  MODEL=google/gemma-4-31B-it FMT=gemma4 MML=8192 JOBS=$E/jobs_gemma.jsonl OUT=$R/gemma bash vigilante/pod/run_jobs.sh || echo "[r28] gemma FAILED" | tee -a $R/pod_gemma.log
fi
echo "[r28] all done ($WHICH)" | tee -a $R/pod_done.log
