#!/usr/bin/env bash
# Talk to the running s1-1 pod (ssh target from launch.log).
#   bash selfknow/s1_persistence/pod_ctl.sh pull            # copy results/ from the pod (tar over ssh)
#   bash selfknow/s1_persistence/pod_ctl.sh push            # copy this folder's code + contexts.json to the pod
#   bash selfknow/s1_persistence/pod_ctl.sh ctl main|pilot|stop
#   bash selfknow/s1_persistence/pod_ctl.sh log             # tail the pod logs
set -euo pipefail
cd "$(dirname "$0")/../.."
E=selfknow/s1_persistence
T=$(grep -o "ssh root@[0-9.]* -p [0-9]*" $E/launch.log | tail -1)
HOST=$(echo $T | awk '{print $2}'); PORT=$(echo $T | awk '{print $4}')
O=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i ~/.ssh/id_ed25519)
case "$1" in
  pull) ssh "${O[@]}" -p $PORT $HOST "cd /root/NEEL && tar -czf - $E/results" | tar -xzf - -C . && echo pulled ;;
  push) tar -czf - $E/*.py $E/contexts.json | ssh "${O[@]}" -p $PORT $HOST "cd /root/NEEL && tar -xzf -" && echo pushed ;;
  ctl) ssh "${O[@]}" -p $PORT $HOST "mkdir -p /root/NEEL/$E/control && touch /root/NEEL/$E/control/$2" && echo "sent $2" ;;
  log) ssh "${O[@]}" -p $PORT $HOST "cd /root/NEEL/$E/results && tail -5 pod.log; for f in */run_*.log; do echo == \$f; grep -E '\[s1\]|\[runner\]|\[render\]|Error|error' \$f | tail -${2:-6}; done" ;;
esac
