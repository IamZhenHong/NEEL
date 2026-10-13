#!/usr/bin/env bash
# R29 on one A100 80GB: vLLM titration (Gemma 4 31B, thinking), then transformers reads + directions (hf_reads.py),
# then thought-gated steering of the test level and B1@closed (src/steer_run.py, project venv).
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r29_titrate
R=$E/results
mkdir -p $R
MODEL=google/gemma-4-31B-it FMT=gemma4 MML=8192 JOBS=$E/jobs_titr.jsonl OUT=$R/titr bash vigilante/pod/run_jobs.sh || { echo "[r29] titration FAILED" | tee -a $R/pod.log; exit 1; }
pkill -9 -f EngineCore; sleep 5
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m vigilante.exp.r29_titrate.hf_reads --titr $R/titr/episodes_agent.jsonl --out $R/hf 2>&1 | tee -a $R/hf.log || { echo "[r29] reads FAILED" | tee -a $R/pod.log; exit 1; }
python -m vigilante.src.steer_run --jobs $R/hf/steer_jobs.jsonl --out $R/hf --layer 30 --batch 8 --max-rounds 10 2>&1 | tee -a $R/steer.log
echo "[r29] all done" | tee -a $R/pod.log
