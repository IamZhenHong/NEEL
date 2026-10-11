#!/usr/bin/env bash
# Version-A pilot generation (eval_affect/PILOT_A_DESIGN.md): Qwen3-32B bf16, 200 jobs, H200.
#   uv run python infra/launch_run.py --name evalaff-pilotA --max-minutes 55 --max-cost 5.0 \
#       --gpus "NVIDIA H200" "NVIDIA H100 NVL" --container-gb 160 --allow-concurrent \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir eval_affect/results/pilot_a -- bash eval_affect/pod/pilot_a.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=eval_affect/results/pilot_a; mkdir -p $OUT
bash eval_affect/pod/setup_vllm.sh 2>&1 | tee -a $OUT/run.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python eval_affect/code/pod_gen.py --jobs eval_affect/materials/pilot_a/jobs.jsonl --out $OUT ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a $OUT/run.log
echo "[pilot_a] pod done" | tee -a $OUT/run.log
