#!/usr/bin/env bash
# Keeps an ssh tunnel localhost:8765 -> pod:8765 alive. Usage: tunnel.sh HOST PORT
while true; do
  ssh -N -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ServerAliveInterval=20 \
      -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -i ~/.ssh/id_ed25519 -L 8765:127.0.0.1:8765 -p "$2" root@"$1"
  echo "[tunnel] exited $(date +%T); reconnecting in 5s"
  sleep 5
done
