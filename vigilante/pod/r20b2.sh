#!/usr/bin/env bash
# R20 stage B2: transformers reads + directions + decision-point steering (project venv, no vLLM), one A100 80GB.
#   NAME=gemma MODEL=google/gemma-4-31B-it FMT=gemma4 LAYERS="20 30 40" PRIMARY=30 BATCH=6 \
#   EXTRA=vigilante/exp/r17_steer/results/dirs.pt bash vigilante/pod/r20b2.sh
# READS / STEER / OUT override the default files (control run: OUT holds a copy of dirs.pt, so no re-extraction).
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${NAME:?}" "${MODEL:?}" "${FMT:?}" "${LAYERS:?}" "${PRIMARY:?}"
H=vigilante/exp/r20_vigilante_dir
READS=${READS:-$H/reads_b2_$NAME.jsonl}; STEER=${STEER:-$H/steer_b2_$NAME.jsonl}
OUT=${OUT:-$H/results/b2_$NAME}; mkdir -p $OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $OUT/run.log
python -m vigilante.exp.r20_vigilante_dir.pod_b2 --model $MODEL --fmt $FMT --reads $READS \
    --steer $STEER --out $OUT --layers $LAYERS --primary $PRIMARY --batch ${BATCH:-6} \
    ${EXTRA:+--extra-dirs $EXTRA} ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a $OUT/run.log
echo "[r20b2] pod done" | tee -a $OUT/run.log
