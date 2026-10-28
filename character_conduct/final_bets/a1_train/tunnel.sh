#!/usr/bin/env bash
# CC-F-A1: keep an ssh tunnel from the laptop's 127.0.0.1:LOCAL (default 8000) to a pod's vLLM at 127.0.0.1:8000,
# reconnecting after drops. Ports: pod S 8000, T1 8001, T2 8002, T3 8003.
#   bash character_conduct/final_bets/a1_train/tunnel.sh HOST PORT [LOCAL]      (stop: touch /tmp/a1_tunnel_stop_LOCAL)
H=${1:?}; P=${2:?}; L=${3:-8000}
rm -f "/tmp/a1_tunnel_stop_$L"
while [ ! -f "/tmp/a1_tunnel_stop_$L" ]; do
  ssh -N -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes -i ~/.ssh/id_ed25519 \
      -L "$L:127.0.0.1:8000" -p "$P" "root@$H"
  echo "[tunnel $L] dropped $(date +%H:%M:%S); reconnecting"; sleep 5
done
