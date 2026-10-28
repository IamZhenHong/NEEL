#!/usr/bin/env bash
# CC-F-A1 (deviation 10): retire a serving replica once its work is done. Waits until every WAIT_LOG (local eval logs)
# says "chunk done", pulls the pod's records (no weights) into results/pod_<POD>/, stops the laptop tunnel on LOCAL,
# terminates the pod (retried until RunPod no longer lists it) and books elapsed x rate to CC-F-A1.
#   bash character_conduct/final_bets/a1_train/finish_replica.sh POD HOST PORT POD_ID RATE START_EPOCH LOCAL [WAIT_LOG ...]
set -uo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
POD=${1:?}; HOST=${2:?}; PORT=${3:?}; ID=${4:?}; RATE=${5:?}; START=${6:?}; LOCAL=${7:?}; shift 7
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 -p "$PORT" "root@$HOST")
log() { echo "[retire $POD] $(date +%H:%M:%S) $*"; }
for W in "$@"; do
  until grep -q "chunk done\|forecasts done\|desk done" "$W" 2>/dev/null; do sleep 60; done
  log "done: $W"
done
mkdir -p "$H/results/pod_$POD"
"${SSH[@]}" "cd /root/NEEL/$H/results && tar cf - serve/pod.log serve/serve.log serve/setup_vllm.log serve/gen train/*.json train/*.log train/DONE_* 2>/dev/null" | tar xf - -C "$H/results/pod_$POD"
"${SSH[@]}" "cd /root && tar cf - *.log 2>/dev/null" | tar xf - -C "$H/results/pod_$POD"
log "records pulled: $(find "$H/results/pod_$POD" -type f | wc -l | tr -d ' ') files"
touch "/tmp/a1_tunnel_stop_$LOCAL"
until uv run python -c "
import sys; sys.path.insert(0, 'infra'); import runpod_guard as g
if any(p['id'] == '$ID' for p in g.account().pods): g.terminate('$ID')
sys.exit(1 if any(p['id'] == '$ID' for p in g.account().pods) else 0)" 2>/dev/null; do log "terminate not confirmed; retrying"; sleep 30; done
MIN=$(( ($(date +%s) - START + 59) / 60 ))
USD=$(python3 -c "import math, sys; print(math.ceil(int(sys.argv[1]) * float(sys.argv[2]) / 60 * 1000) / 1000)" "$MIN" "$RATE")
uv run python -c "
import sys; from character_conduct.common.ledger import ledger_add
ledger_add('CC-F-A1', 'gpu', float(sys.argv[1]), sys.argv[2])" "$USD" "cc-f-a1-$(echo "$POD" | tr 'A-Z' 'a-z') $ID (serving replica; $MIN min x \$$RATE/h)"
log "terminated $ID; booked \$$USD ($MIN min x \$$RATE/h)"
