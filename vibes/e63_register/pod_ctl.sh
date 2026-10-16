#!/usr/bin/env bash
# Control of the pod's hold step (pod.sh). Host/port come from the launcher log ("[pod] ssh root@H -p P").
#   bash vibes/e63_register/pod_ctl.sh <launch log> pull-pilot   # scp results/pod/{pilot,engine}/* (read only)
#   bash vibes/e63_register/pod_ctl.sh <launch log> go           # refuses unless the tag e63-freeze exists
#   bash vibes/e63_register/pod_ctl.sh <launch log> tail         # last lines of the pod log
set -uo pipefail
cd "$(dirname "$0")/../.."
D=vibes/e63_register
read -r H P < <(grep -oE "ssh root@[0-9.]+ -p [0-9]+" "$1" | tail -1 | sed -E 's/ssh root@([0-9.]+) -p ([0-9]+)/\1 \2/')
O=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=30 -i ~/.ssh/id_ed25519)
case "$2" in
  pull-pilot)
    mkdir -p $D/results/pod/pilot $D/results/pod/engine
    scp "${O[@]}" -P "$P" "root@$H:/root/NEEL/$D/results/pod/pilot/*" $D/results/pod/pilot/ && \
    scp "${O[@]}" -P "$P" "root@$H:/root/NEEL/$D/results/pod/engine/*" $D/results/pod/engine/ && \
    scp "${O[@]}" -P "$P" "root@$H:/root/NEEL/$D/results/pod/pod.log" $D/results/pod/pod_pilot.log && ls -la $D/results/pod/pilot ;;
  go)
    git rev-parse -q --verify refs/tags/e63-freeze >/dev/null || { echo "no e63-freeze tag; refusing"; exit 1; }
    ssh "${O[@]}" -p "$P" "root@$H" "echo 'e63-freeze $(git rev-parse --short e63-freeze)' > /root/NEEL/$D/GO && cat /root/NEEL/$D/GO" ;;
  tail)
    ssh "${O[@]}" -p "$P" "root@$H" "tail -n ${3:-20} /root/NEEL/$D/results/pod/pod.log" ;;
esac
