#!/usr/bin/env bash
# jl2 + jl3 on one A100 80GB (Qwen3.6-27B, Neuronpedia J-lens / J++ lens): one model load, both cards.
# jl3's outputs land in jlens/e2_realism/results/jl3 (the launcher pulls one directory) and are moved to
# jlens/e3_hallucination/results after the pull.
#   caffeinate -i uv run python infra/launch_run.py --name jl23-1 --allow-concurrent --max-minutes 55 --max-cost 2 \
#       --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" --container-gb 160 \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir jlens/e2_realism/results -- bash jlens/e2_realism/pod.sh
# Local CPU test of every code path: uv run python -m jlens.e2_realism.pod_jl23 --tiny --limit 6 --out2 <d> --out3 <d>
set -uo pipefail
cd "$(dirname "$0")/../.."
R=jlens/e2_realism/results
mkdir -p $R/jl3
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
python -m jlens.e2_realism.pod_jl23 --out2 $R --out3 $R/jl3 2>&1 | grep --line-buffered -v "Loading weights\|Fetching" | tee -a $R/pod.log
echo "[jl23] job exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
