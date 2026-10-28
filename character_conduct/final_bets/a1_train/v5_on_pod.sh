#!/usr/bin/env bash
# CC-F-A1 V5 s1 on a training pod after its block (deviation 7/8b). Run on the pod with setsid nohup:
#   bash character_conduct/final_bets/a1_train/v5_on_pod.sh AFTER_NAME V5_SHA
# Waits for "-> AFTER_NAME" in trainseq.log, sets /root/PAUSE (pod_serve.sh stops its server), waits for the GPU to
# free, then trains V5:1 from data/train_versions_v5.jsonl.gz with KEEP_PAUSE=1 (the pod is terminated afterwards).
set -uo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
AFTER=${1:?}; SHA=${2:?}
until grep -q -- "-> $AFTER " "$H/results/train/trainseq.log" 2>/dev/null; do sleep 20; done
touch /root/PAUSE; sleep 30
for i in $(seq 1 40); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 3000 ] && break
  pkill -f "VLLM::EngineCore"; pkill -f "vllm.entrypoints"; sleep 15
done
echo "[a1] V5 start $(date -u +%H:%M), GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
DATA=$H/data/train_versions_v5.jsonl.gz KEEP_PAUSE=1 bash "$H/pod_train_seq2.sh" "$H/results/train" "$SHA" V5:1
