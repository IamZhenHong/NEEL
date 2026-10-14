#!/usr/bin/env bash
# R45 on one A100 80GB: Qwen3.6-27B, thinking off, C0 / C4 / C5 (jobs_qwen.jsonl), as R27's Qwen setup.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=vigilante/exp/r45_initiative_qwen
MODEL=Qwen/Qwen3.6-27B FMT=qwen MML=40960 JOBS=$E/jobs_qwen.jsonl OUT=$E/results bash vigilante/pod/run_jobs.sh
