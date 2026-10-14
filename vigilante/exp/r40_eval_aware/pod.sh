#!/usr/bin/env bash
# R40 on one A100 80GB (Qwen3.6-27B): behaviour episodes (vLLM), 0-100 ratings (vLLM), last-token reads (transformers).
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r40_eval_aware
R=$E/results
mkdir -p $R
MODEL=Qwen/Qwen3.6-27B FMT=qwen MML=40960 JOBS=$E/jobs_qwen.jsonl OUT=$R bash vigilante/pod/run_jobs.sh || echo "[r40] behaviour FAILED" | tee -a $R/pod.log
pkill -9 -f EngineCore; sleep 5
PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 /root/vllm/bin/python -m vigilante.exp.r40_eval_aware.pod_r40 --stage ratings --out $R 2>&1 | tee -a $R/pod.log || echo "[r40] ratings FAILED" | tee -a $R/pod.log
pkill -9 -f EngineCore; sleep 5
python -m vigilante.exp.r40_eval_aware.pod_r40 --stage reads --out $R 2>&1 | tee -a $R/pod.log || echo "[r40] reads FAILED" | tee -a $R/pod.log
echo "[r40] all done" | tee -a $R/pod.log
