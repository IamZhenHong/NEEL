#!/usr/bin/env bash
# R21 on one A100 80GB: (1) record the 16-chat histories and run the short grid; (2) pick a history per case and
# write jobs_long.jsonl; (3) run the long cells (the model loads twice).
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r21_approval_fix
R=$E/results
mkdir -p $R
export MODEL=google/gemma-4-31B-it FMT=gemma4 MML=8192
cat $E/jobs_hist.jsonl | sed 's/"group": "agent"/"group": "chat"/' > $E/jobs_stage1.jsonl
cat $E/jobs_a.jsonl >> $E/jobs_stage1.jsonl
JOBS=$E/jobs_stage1.jsonl OUT=$R/a bash vigilante/pod/run_jobs.sh
/root/vllm/bin/python -m vigilante.exp.r21_approval_fix.make_long --hist $R/a/episodes_chat.jsonl --out $R/long 2>&1 | tee -a $R/long_prep.log
cp $E/jobs_long.jsonl $R/long/jobs_long.jsonl
JOBS=$E/jobs_long.jsonl OUT=$R/long bash vigilante/pod/run_jobs.sh
echo "[r21] all done" | tee -a $R/long_prep.log
