#!/usr/bin/env bash
# Talk to the running s2 pod (host/port from launch.log).
#   bash selfknow/s2_role_wins/remote.sh pull-pilot      # scp results/<model>/pilot_<model>/ here
#   bash selfknow/s2_role_wins/remote.sh push-main       # upload jobs/main_{qwen,gemma}.jsonl into the pod inboxes
#   bash selfknow/s2_role_wins/remote.sh stop <qwen|gemma>   # write STOP into an inbox
#   bash selfknow/s2_role_wins/remote.sh tail            # last lines of the pod job log
set -euo pipefail
cd "$(dirname "$0")/../.."
E=selfknow/s2_role_wins
L=$(grep -m1 "\[pod\] ssh root@" $E/launch.log)
HOST=$(echo "$L" | sed -E 's/.*root@([^ ]+) -p ([0-9]+).*/\1/')
PORT=$(echo "$L" | sed -E 's/.*root@([^ ]+) -p ([0-9]+).*/\2/')
O=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i ~/.ssh/id_ed25519)
case "$1" in
  pull-pilot)
    for m in gemma qwen; do
      mkdir -p $E/results/$m
      scp -r "${O[@]}" -P $PORT root@$HOST:/root/NEEL/$E/results/$m/pilot_$m $E/results/$m/ 2>/dev/null && echo "pulled $m pilot" || echo "no $m pilot yet"
    done ;;
  push-main)
    for m in qwen gemma; do
      scp "${O[@]}" -P $PORT $E/jobs/main_$m.jsonl root@$HOST:/root/s2_inbox/$m/main_$m.jsonl.part
      ssh "${O[@]}" -p $PORT root@$HOST "mv /root/s2_inbox/$m/main_$m.jsonl.part /root/s2_inbox/$m/main_$m.jsonl && sha256sum /root/s2_inbox/$m/main_$m.jsonl"
    done
    shasum -a 256 $E/jobs/main_*.jsonl ;;
  stop)
    ssh "${O[@]}" -p $PORT root@$HOST "touch /root/s2_inbox/$2/STOP && ls /root/s2_inbox/$2" ;;
  tail)
    ssh "${O[@]}" -p $PORT root@$HOST "tail -n ${2:-20} /root/job.log" ;;
esac
