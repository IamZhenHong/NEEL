#!/usr/bin/env bash
# e70 pilot pod (before neurons-freeze): gate (determinism, greedy claim token), e70 attribution on the 8 pilot
# transcripts (10%), ablation timing, base claim rate. Launch from the repo root (see README "Runs").
set -uo pipefail
cd "$(dirname "$0")/.."
R=neurons/results/pilot
mkdir -p $R
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
df -h /root | tail -1 | tee -a $R/pod.log
python -m neurons.pod_attr --kind qwen36 --out $R --stages gate e70 abltime claimbase --pilot 2>&1 \
  | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a $R/pod.log
echo "[neurons] pilot exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
du -sh $R | tee -a $R/pod.log
