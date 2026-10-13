#!/usr/bin/env bash
# R34: build v_traj / v_dec (step 1), then steer R33's B1 items with them (step 2, pod_b2; reuses R20's dirs.pt in OUT).
#   bash vigilante/pod/r34.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
H=vigilante/exp/r34_traj
OUT=$H/results; mkdir -p $OUT
R20=vigilante/exp/r20_vigilante_dir/results/b2_gemma
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $OUT/run.log
if [ ! -f $OUT/dirs_r34.pt ]; then
  python -m vigilante.exp.r34_traj.pod_dirs --model google/gemma-4-31B-it --reads $H/reads_r34.jsonl --out $OUT \
      --vig-dirs $OUT/dirs.pt --placebo $R20/placebo_dirs.pt 2>&1 | tee -a $OUT/run.log
fi
python -m vigilante.exp.r20_vigilante_dir.pod_b2 --model google/gemma-4-31B-it --fmt gemma4 --reads $H/reads_r34.jsonl \
    --steer $H/steer_r34.jsonl --out $OUT --layers 30 --primary 30 --batch 8 --extra-dirs $OUT/dirs_r34.pt 2>&1 \
    | tee -a $OUT/run.log
echo "[r34] pod done" | tee -a $OUT/run.log
