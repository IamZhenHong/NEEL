#!/usr/bin/env bash
# c2 rerun on one A100 80GB (README Amendment 9): Gemma smoke, Qwen smoke + pre-pass, Gemma traces on kept items, Qwen
# build + gate + factorial. One model per process; a failed stage stops the pipeline.
#   uv run python infra/launch_run.py --name c2-3 --allow-concurrent --max-minutes 60 --max-cost 1.8 \
#       --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters $(slim excludes: README Amendment 10) \
#       --results-dir cot/c2_provenance/results -- bash cot/c2_provenance/pod.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
R=cot/c2_provenance/results
mkdir -p $R
export C2_T0=$(date +%s)
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
for S in gemma_smoke qwen_a gemma qwen_b; do
  /root/vllm/bin/python -m cot.c2_provenance.pod_c2 --stage $S --out $R 2>&1 \
    | grep --line-buffered -v "Loading safetensors\|Fetching\|it/s\]" | tee -a $R/pod.log
  rc=${PIPESTATUS[0]}
  echo "[c2] stage $S exit $rc" | tee -a $R/pod.log
  if [ "$rc" != "0" ]; then echo "[c2] stopping after failed stage $S" | tee -a $R/pod.log; exit 1; fi
done
echo "[c2] all stages done" | tee -a $R/pod.log
