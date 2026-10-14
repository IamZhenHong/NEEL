#!/usr/bin/env bash
# move_run.sh RUN NEWPORT -- <attack args>: waits until the server on NEWPORT is healthy, then restarts RUN against it
# (resumes from its checkpoint). Used when a pod reaches its time cap.
set -u
RUN=$1; LP=$2; shift 3
cd "$(dirname "$0")/../.."
until SA_URL=http://127.0.0.1:$LP .venv/bin/python -c "from state_attack.src import client as C; C.health()" 2>/dev/null; do sleep 15; done
pkill -f "attack --run $RUN " ; sleep 3
echo "--- moved to :$LP $(date +%T)" >> "state_attack/logs/$RUN.log"
SA_URL=http://127.0.0.1:$LP nohup uv run --with anthropic python -u -m state_attack.src.attack --run "$RUN" "$@" >> "state_attack/logs/$RUN.log" 2>&1 &
echo "[move] $RUN -> :$LP $(date +%T)"
