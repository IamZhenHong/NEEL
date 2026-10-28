#!/usr/bin/env bash
# CC-F-A1: keep an ssh tunnel from the laptop's 127.0.0.1:8000 to pod S's vLLM, reconnecting after drops.
#   bash character_conduct/final_bets/a1_train/tunnel.sh HOST PORT      (stop: touch /tmp/a1_tunnel_stop)
H=${1:?}; P=${2:?}
rm -f /tmp/a1_tunnel_stop
while [ ! -f /tmp/a1_tunnel_stop ]; do
  ssh -N -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 \
      -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ExitOnForwardFailure=yes -i ~/.ssh/id_ed25519 \
      -L 8000:127.0.0.1:8000 -p "$P" "root@$H"
  echo "[tunnel] dropped $(date +%H:%M:%S); reconnecting"; sleep 5
done
