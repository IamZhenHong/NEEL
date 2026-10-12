#!/usr/bin/env bash
# Generic: run a jobs.jsonl with vLLM on one A100 80GB.  JOBS=<jobs.jsonl> OUT=<results dir> [LIMIT=n] [QGROUPS=...]
# Other models: MODEL=google/gemma-4-31B-it FMT=gemma4 MML=16384
#   uv run python infra/launch_run.py --name vig-<exp> --max-minutes M --max-cost C \
#       --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir <OUT> -- env JOBS=<jobs> OUT=<OUT> bash vigilante/pod/run_jobs.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${JOBS:?set JOBS}" "${OUT:?set OUT}"
mkdir -p "$OUT"
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a "$OUT/run.log"
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m vigilante.src.pod_run --jobs "$JOBS" --out "$OUT" ${LIMIT:+--limit $LIMIT} ${QGROUPS:+--groups $QGROUPS} ${MODEL:+--model $MODEL} ${FMT:+--fmt $FMT} ${MML:+--max-model-len $MML} 2>&1 | tee -a "$OUT/run.log"
echo "[run_jobs] pod done" | tee -a "$OUT/run.log"
