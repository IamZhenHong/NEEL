#!/usr/bin/env bash
# readers stage B (main reads, after readers-freeze): AO + P(Yes) + J-lens for e65, e66, e67, e68 (each written as it
# finishes, e65 first), then the NLA phase. NLA_MAX_NEW is fixed in README (A1).
set -uo pipefail
cd "$(dirname "$0")/.."
R=readers/results/main
mkdir -p $R
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
python -m readers.pod_read reads --out $R --sets e65 e66 e67 e68 --nla-max-new ${NLA_MAX_NEW:-200} 2>&1 \
  | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a $R/pod.log
echo "[readers] main exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
rm -f $R/vec42_*.npz
du -sh $R | tee -a $R/pod.log
