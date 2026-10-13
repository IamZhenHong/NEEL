#!/usr/bin/env bash
# R23-R26 on Qwen3.6-27B (one A100 80GB, model loads three times):
# (1) R23 history blocks + R23 fresh cells + R24 (Qwen identities) + R26 (Qwen); (2) R25 generations;
# (3) pick history samples and write the long jobs; (4) R23 long cells, batched by length.
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r23_long_mech
R=$E/results/qwen
mkdir -p $R
export MODEL=Qwen/Qwen3.6-27B FMT=qwen MML=40960
cat $E/jobs_qwen_s1.jsonl vigilante/exp/r24_family_name/jobs_qwen.jsonl vigilante/exp/r26_initiative/jobs_qwen.jsonl > $E/jobs_qwen_stage1.jsonl
JOBS=$E/jobs_qwen_stage1.jsonl OUT=$R/s1 bash vigilante/pod/run_jobs.sh || echo "[r23q] stage 1 FAILED" | tee -a $R/pod.log
/root/vllm/bin/python -m vigilante.exp.r25_theatrical_big.gen --model Qwen/Qwen3.6-27B --out $R/r25 2>&1 | tee -a $R/r25.log || echo "[r23q] r25 FAILED" | tee -a $R/pod.log
/root/vllm/bin/python -m vigilante.exp.r23_long_mech.make_long --hist $R/s1/episodes_chat.jsonl --out $R/long 2>&1 | tee -a $R/long_prep.log
cp $E/jobs_qwen_long.jsonl $R/long/jobs_qwen_long.jsonl
JOBS=$E/jobs_qwen_long.jsonl OUT=$R/long bash vigilante/pod/run_jobs.sh || echo "[r23q] long FAILED" | tee -a $R/pod.log
echo "[r23q] all done" | tee -a $R/pod.log
