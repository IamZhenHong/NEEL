#!/usr/bin/env bash
# Pull a compressed snapshot of each running P1 main pod's results every PERIOD seconds, replacing the
# previous one (atomic mv), so a failed end-of-run pull or a launcher crash on a full laptop disk cannot
# lose the data. Skips (and says so) when local free space is below FLOOR_MB. Exits when no P1 pod remains.
cd /Users/zhenhongseng/projects/ai_safety/NEEL || exit 1
PERIOD=${PERIOD:-1200}; FLOOR_MB=${FLOOR_MB:-40}
OUT=mas_pilot/p1_baserate/snapshots
while :; do
  alive=0
  for M in gemma qwen; do
    LOG=mas_pilot/p1_baserate/logs/main_${M}_launch1.log
    grep -q "terminated " "$LOG" 2>/dev/null && continue
    alive=1
    free_mb=$(df -m . | awk 'NR==2{print $4}')
    if [ "$free_mb" -lt "$FLOOR_MB" ]; then echo "$(date +%H:%M) SKIP $M snapshot: only ${free_mb} MiB free"; continue; fi
    HP=$(grep -oE "ssh root@[0-9.]+ -p [0-9]+" "$LOG" | tail -1); HOST=$(echo "$HP" | awk '{print $2}' | cut -d@ -f2); PORT=$(echo "$HP" | awk '{print $4}')
    if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i ~/.ssh/id_ed25519 \
         -o ConnectTimeout=20 -p "$PORT" "root@$HOST" "tar -czf - -C /root/NEEL mas_pilot/p1_baserate/results/main_${M}" \
         > "$OUT/.main_${M}.tmp" 2>/dev/null && gzip -t "$OUT/.main_${M}.tmp" 2>/dev/null; then
      mv -f "$OUT/.main_${M}.tmp" "$OUT/main_${M}_latest.tar.gz"
      echo "$(date +%H:%M) $M snapshot $(du -h "$OUT/main_${M}_latest.tar.gz" | cut -f1), free ${free_mb} MiB"
    else
      rm -f "$OUT/.main_${M}.tmp"; echo "$(date +%H:%M) $M snapshot FAILED (kept previous), free ${free_mb} MiB"
    fi
  done
  [ "$alive" -eq 0 ] && { echo "$(date +%H:%M) no P1 main pod left; snapshot loop done"; exit 0; }
  sleep "$PERIOD"
done
