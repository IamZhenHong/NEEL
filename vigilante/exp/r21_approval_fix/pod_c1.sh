#!/usr/bin/env bash
# R21 deviation (C1 long only): the first C1 histories all refunded a cancelled $198 order (chat 9), so none was valid;
# chat 9 now asks for a colour swap instead (no money). Record 3 new C1 histories, then run the 30 C1 long cells.
set -euo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r21_approval_fix
R=$E/results/c1fix
mkdir -p $R
export MODEL=google/gemma-4-31B-it FMT=gemma4 MML=8192
JOBS=$E/jobs_hist_c1.jsonl OUT=$R/hist bash vigilante/pod/run_jobs.sh
/root/vllm/bin/python -m vigilante.exp.r21_approval_fix.make_long --hist $R/hist/episodes_agent.jsonl --out $R/long 2>&1 | tee -a $R/long_prep.log
cp $E/jobs_long.jsonl $R/long/jobs_long.jsonl
JOBS=$E/jobs_long.jsonl OUT=$R/long bash vigilante/pod/run_jobs.sh
echo "[r21c1] all done" | tee -a $R/long_prep.log
