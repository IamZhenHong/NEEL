#!/usr/bin/env bash
# Section-0 quick experiments (vigilante/exp/q0_quick): Qwen3.6-27B bf16 on one A100 80GB, groups chat -> agent -> think.
#   uv run python infra/launch_run.py --name vig-q0 --max-minutes 170 --max-cost 6.0 \
#       --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir vigilante/exp/q0_quick/results -- bash vigilante/pod/q0_quick.sh
# Smoke: prefix the job with LIMIT=3 (first 3 jobs per group). One group on its own pod:
#   ... -- env QGROUPS=think OUT=vigilante/exp/q0_quick/results_think bash vigilante/pod/q0_quick.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-vigilante/exp/q0_quick/results}; mkdir -p $OUT
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $OUT/run.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m vigilante.src.pod_run --jobs vigilante/exp/q0_quick/jobs.jsonl --out $OUT ${LIMIT:+--limit $LIMIT} ${QGROUPS:+--groups $QGROUPS} 2>&1 | tee -a $OUT/run.log
echo "[q0] pod done" | tee -a $OUT/run.log
