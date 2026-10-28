#!/usr/bin/env bash
# CC-F-A1: train on a serving replica (deviation 8c): set /root/PAUSE (pod_serve.sh stops its server), wait for the GPU
# to free, train LIST with pod_train_seq2.sh (STEPS from results/train/steps.json), which removes PAUSE at the end so the
# server restarts with every adapter in /root/loras.txt. Run on the pod with setsid nohup:
#   bash character_conduct/final_bets/a1_train/train_on_replica.sh "V2:2" ce676243...
set -uo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
LIST=${1:?}; SHA=${2:?}
touch /root/PAUSE; sleep 30
for i in $(seq 1 40); do
  [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -lt 3000 ] && break
  pkill -f "VLLM::EngineCore"; pkill -f "vllm.entrypoints"; sleep 15
done
echo "[a1] replica training $LIST start $(date -u +%H:%M), GPU $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
bash "$H/pod_train_seq2.sh" "$H/results/train" "$SHA" $LIST
