#!/usr/bin/env bash
# s3 Part 2 on one A100 80GB: Gemma 4 12B, then Qwen3.6-27B, each its own vLLM process.  MODE=pilot|main
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${MODE:?set MODE=pilot or main}"
E=selfknow/s3_saydo_causes
R=$E/results/$MODE
mkdir -p $R
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
for M in gemma qwen; do
  /root/vllm/bin/python $E/pod_s3.py --model $M --mode $MODE --out $R/$M 2>&1 | tee -a $R/pod.log \
    || echo "[s3] $M FAILED" | tee -a $R/pod.log
  rm -rf /root/hf/hub/models--google--gemma-4-12B-it   # free container disk before the next model
done
echo "[s3] pod done" | tee -a $R/pod.log
