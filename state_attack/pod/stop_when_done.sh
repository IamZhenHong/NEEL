#!/usr/bin/env bash
# stop_when_done.sh RUN HOST PORT LOCALPORT: when RUN logs "all waves done", stop its pod (STOP file + kill server;
# the launcher then pulls results and terminates it) and close the tunnel.
RUN=$1; H=$2; P=$3; LP=$4
cd "$(dirname "$0")/../.."
until grep -q "all waves done" "state_attack/logs/$RUN.log"; do sleep 20; done
state_attack/pod/podssh.sh "$H" "$P" 'touch /root/STOP; for q in $(pgrep -f "^python -u -m state_attack.src.server"); do kill $q; done'
sleep 5; pkill -f "tunnel.sh $H $P"; pkill -f "ssh -N.*-L $LP:"
echo "[stop] $RUN pod $H:$P stopped $(date +%T)"
