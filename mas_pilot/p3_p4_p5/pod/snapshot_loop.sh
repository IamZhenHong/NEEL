#!/usr/bin/env bash
# Every PERIOD s, pull a compressed snapshot of a running pod's results dir (atomic replace); exit when the
# launch log says the pod was terminated. Args: LAUNCH_LOG POD_RESULTS_REL LOCAL_SNAPSHOT
cd /Users/zhenhongseng/projects/ai_safety/NEEL || exit 1
LOG="$1"; REL="$2"; SNAP="$3"; PERIOD=${PERIOD:-1200}
while :; do
  grep -q "terminated " "$LOG" 2>/dev/null && { echo "$(date +%H:%M) pod terminated; snapshot loop done"; exit 0; }
  HP=$(grep -oE "ssh root@[0-9.]+ -p [0-9]+" "$LOG" | tail -1)
  if [ -n "$HP" ]; then
    HOST=$(echo "$HP" | awk '{print $2}' | cut -d@ -f2); PORT=$(echo "$HP" | awk '{print $4}')
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i ~/.ssh/id_ed25519 \
        -o ConnectTimeout=20 -p "$PORT" "root@$HOST" "test -d /root/NEEL/$REL && tar -czf - -C /root/NEEL $REL" > "$SNAP.tmp" 2>/dev/null \
        && gzip -t "$SNAP.tmp" 2>/dev/null; then mv -f "$SNAP.tmp" "$SNAP"; echo "$(date +%H:%M) snapshot $(du -h "$SNAP" | cut -f1)"
    else rm -f "$SNAP.tmp"; echo "$(date +%H:%M) snapshot skipped/failed"; fi
  fi
  sleep "$PERIOD"
done
