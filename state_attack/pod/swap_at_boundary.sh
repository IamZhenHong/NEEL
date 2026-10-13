#!/usr/bin/env bash
# swap_at_boundary.sh RUN HOST PORT LOCALPORT -- <attack args>
# Waits until RUN logs "wave N done", stops its client, deploys server.py/client.py to the pod, restarts the server
# (supervisor loop restarts it), waits for health, and restarts the client with the same arguments (it resumes at the
# next wave: finished waves and notes are on disk).
set -u
RUN=$1; HOST=$2; PORT=$3; LP=$4; shift 5
cd "$(dirname "$0")/../.."
LOG=state_attack/logs/$RUN.log
n0=$(grep -c "done in" "$LOG")
until [ "$(grep -c "done in" "$LOG")" -gt "$n0" ]; do sleep 10; done
pkill -f "attack --run $RUN " ; sleep 2
SSHO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -i $HOME/.ssh/id_ed25519"
scp -q $SSHO -P "$PORT" state_attack/src/server.py state_attack/src/client.py root@"$HOST":/root/NEEL/state_attack/src/
ssh $SSHO -p "$PORT" root@"$HOST" 'for p in $(pgrep -f "^python -u -m state_attack.src.server"); do kill $p; done; echo killed'
sleep 25
until SA_URL=http://127.0.0.1:$LP .venv/bin/python -c "from state_attack.src import client as C; C.health()" 2>/dev/null; do sleep 10; done
mv "$LOG" "state_attack/logs/${RUN}_part$(date +%H%M).log"
SA_URL=http://127.0.0.1:$LP nohup uv run --with anthropic python -u -m state_attack.src.attack --run "$RUN" "$@" > "$LOG" 2>&1 &
echo "[swap] $RUN restarted on $HOST at $(date +%T)"
