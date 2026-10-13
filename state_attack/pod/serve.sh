#!/usr/bin/env bash
# Keeps the target server up until /root/STOP exists (restart after a crash or a code update: kill the python PID).
cd /root/NEEL
mkdir -p state_attack/results/pod
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a state_attack/results/pod/server.log
while [ ! -f /root/STOP ]; do
  python -u -m state_attack.src.server --port 8765 2>&1 | tee -a state_attack/results/pod/server.log
  echo "[serve] server exited; restarting in 5s" | tee -a state_attack/results/pod/server.log
  sleep 5
done
echo "[serve] STOP seen; exiting" | tee -a state_attack/results/pod/server.log
