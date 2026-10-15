#!/usr/bin/env bash
# c2 on one A100 80GB: Gemma 4 31B traces (process 1), then Qwen3.6-27B pre-pass, wait for GO, factorial (process 2).
#   uv run python infra/launch_run.py --name c2-1 --allow-concurrent --max-minutes 66 --max-cost 2.0 \
#       --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
#       --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#       --results-dir cot/c2_provenance/results -- bash cot/c2_provenance/pod.sh
# Local CPU test: see pod_c2.py (--tiny).
set -uo pipefail
cd "$(dirname "$0")/../.."
E=cot/c2_provenance
R=$E/results
mkdir -p $R $E/run
rm -f $E/run/GO $E/run/STOP
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
if [ ! -s $R/prepass_gemma.jsonl ]; then
  /root/vllm/bin/python -m cot.c2_provenance.pod_c2 --stage gemma --out $R 2>&1 | grep --line-buffered -v "Loading safetensors\|Fetching\|it/s\]" | tee -a $R/pod.log
  echo "[c2] gemma stage exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
  rm -rf /root/hf/hub/models--google--gemma-4-31B-it   # free container disk before Qwen downloads
fi
/root/vllm/bin/python -m cot.c2_provenance.pod_c2 --stage qwen --out $R --wait-minutes ${C2_WAIT_MIN:-28} 2>&1 | grep --line-buffered -v "Loading safetensors\|Fetching\|it/s\]" | tee -a $R/pod.log
echo "[c2] qwen stage exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
