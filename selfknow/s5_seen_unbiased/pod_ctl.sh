#!/usr/bin/env bash
# Hand control of the s5-2 hold step (pod_all.sh). Host/port come from launch_s5-2.log ("[pod] ssh root@H -p P").
#   bash selfknow/s5_seen_unbiased/pod_ctl.sh pull-pilot      # scp results/pilot/* from the pod (read only)
#   bash selfknow/s5_seen_unbiased/pod_ctl.sh go <cue> <min>   # refuses unless tag s5-freeze exists
set -uo pipefail
cd "$(dirname "$0")/../.."
D=selfknow/s5_seen_unbiased
read -r H P < <(grep -oE "ssh root@[0-9.]+ -p [0-9]+" $D/launch_s5-2.log | tail -1 | sed -E 's/ssh root@([0-9.]+) -p ([0-9]+)/\1 \2/')
O=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=30 -i ~/.ssh/id_ed25519)
case "$1" in
  pull-pilot)
    mkdir -p $D/results/pilot
    scp "${O[@]}" -P "$P" "root@$H:/root/NEEL/$D/results/pilot/*" $D/results/pilot/ && ls -la $D/results/pilot ;;
  go)
    git rev-parse -q --verify refs/tags/s5-freeze >/dev/null || { echo "no s5-freeze tag; refusing"; exit 1; }
    ssh "${O[@]}" -p "$P" "root@$H" "echo '$2 $3' > /root/NEEL/$D/GO && cat /root/NEEL/$D/GO" ;;
esac
