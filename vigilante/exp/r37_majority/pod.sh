#!/usr/bin/env bash
# R35 + R36 + R37 on one A100 80GB.  WHICH=gemma: R35 gen (Gemma 4 31B), then R36+R37 jobs (Gemma 4 31B).
#                                     WHICH=qwen:  R35 gen (Qwen3-14B), then R36+R37 jobs (Qwen3.6-27B).
set -uo pipefail
cd "$(dirname "$0")/../../.."
R=vigilante/exp/r37_majority/results/$WHICH
mkdir -p $R
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/run.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
if [ "$WHICH" = gemma ]; then M35=google/gemma-4-31B-it; M=google/gemma-4-31B-it; FMT=gemma4; MML=8192
else M35=Qwen/Qwen3-14B; M=Qwen/Qwen3.6-27B; FMT=qwen; MML=40960; fi
/root/vllm/bin/python -m vigilante.exp.r35_framing.gen --model $M35 --out $R/r35 2>&1 | tee -a $R/r35.log || echo "[r35] $WHICH FAILED" | tee -a $R/run.log
pkill -9 -f EngineCore; sleep 5
cat vigilante/exp/r36_levers/jobs_$WHICH.jsonl vigilante/exp/r37_majority/jobs_$WHICH.jsonl > /root/jobs_r3637.jsonl
MODEL=$M FMT=$FMT MML=$MML JOBS=/root/jobs_r3637.jsonl OUT=$R bash vigilante/pod/run_jobs.sh || echo "[r36/37] $WHICH FAILED" | tee -a $R/run.log
echo "[r35-37] all done ($WHICH)" | tee -a $R/run.log
