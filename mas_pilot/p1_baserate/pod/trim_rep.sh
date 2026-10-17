#!/usr/bin/env bash
# A3 rev trim: stop ONE rep in ALL arms on the running pod (uniform across arms, latest rep first).
# Kills that rep's EC harness processes (cli `--start-index N` and its runner child on repNNN's manifest);
# episodes already saved stay on disk and are pulled with the rest. Touches only the pod named in the log.
# Usage: bash trim_rep.sh <launch log> <rep number>
set -euo pipefail
LOG="$1"; REP="$2"; R3=$(printf %03d "$REP")
HP=$(grep -oE "ssh root@[0-9.]+ -p [0-9]+" "$LOG" | tail -1)
HOST=$(echo "$HP" | awk '{print $2}' | cut -d@ -f2); PORT=$(echo "$HP" | awk '{print $4}')
POD=$(grep -oE "pod [a-z0-9]{10,} " "$LOG" | head -1 | awk '{print $2}')
echo "[trim] pod $POD ($HOST:$PORT): stopping rep$R3 in all arms at $(date -u +%H:%M:%S)"
ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i ~/.ssh/id_ed25519 \
    -o ConnectTimeout=30 -p "$PORT" "root@$HOST" "
  pgrep -af 'experiments' | grep -E -- '(--start-index $REP( |\$)|rep${R3}_sampled_manifest)' || true
  pkill -TERM -f 'rep${R3}_sampled_manifest' || true
  pkill -TERM -f -- '--start-index $REP ' || true
  sleep 3
  echo '[trim] remaining harness processes:'; pgrep -af 'experiments' | grep -oE -- '--start-index [0-9]+|rep[0-9]{3}_sampled_manifest' | sort | uniq -c
  for f in /root/NEEL/mas_pilot/p1_baserate/results/*/*/rep${R3}.log; do
    echo \"[trim] \$f: \$(grep -c 'Saved incremental results after' \$f) episodes saved\"; done"
