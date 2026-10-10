#!/usr/bin/env bash
# Generic generation pod job: JOBS=<jobs.jsonl> OUT=<results dir> [CAP=<thinking cap>].
#   uv run python infra/launch_run.py --name socfb-dev --max-minutes 45 --max-cost 2.00 --container-gb 80 \
#       --gpus "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" "NVIDIA L40S" \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir social_feedback/results/dev_v1 \
#       -- env JOBS=social_feedback/materials/jobs/dev_v1/jobs.jsonl OUT=social_feedback/results/dev_v1 bash social_feedback/pod/generate.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${JOBS:?set JOBS}"; : "${OUT:?set OUT}"
mkdir -p $OUT
bash social_feedback/pod/setup_vllm.sh 2>&1 | tee -a $OUT/run.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0   # see calib.sh
/root/vllm/bin/python -m social_feedback.src.pod_generate --jobs $JOBS --out $OUT ${CAP:+--thinking-cap $CAP} 2>&1 | tee -a $OUT/run.log
echo "[generate] pod done" | tee -a $OUT/run.log
