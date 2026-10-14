#!/usr/bin/env bash
# R41 stage G on one A100 80GB: Gemma 4 31B, thinking on, C0 / C4 / C5 (jobs_gemma.jsonl).
set -uo pipefail
cd "$(dirname "$0")/../.."
E=vigilante/exp/r41_initiative
MODEL=google/gemma-4-31B-it FMT=gemma4 MML=12288 JOBS=$E/jobs_gemma.jsonl OUT=$E/results/gemma bash vigilante/pod/run_jobs.sh
