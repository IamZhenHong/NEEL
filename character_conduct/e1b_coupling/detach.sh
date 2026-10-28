#!/usr/bin/env bash
# CC-E1b detached run (note G): survive a laptop network drop. Start launch_habit.sh POD in the background with its
# output in OUTFILE, then run this alongside it:
#   bash character_conduct/e1b_coupling/detach.sh POD OUTFILE MAXMIN
# 1. waits for the launcher's "[pod] running (detached)" line, then kills launch_habit.sh POD and every
#    launch_run.py --name cc-e1b-hPOD process (so neither a lost connection nor the time cap can terminate the pod,
#    and nothing books an upper bound);
# 2. polls /root/job.exit over ssh every 60 s, riding out any number of failed polls, until launch + MAXMIN + 35 min
#    (the pod's boot self-destruct is MAXMIN + 30, the backstop);
# 3. pulls results/hPOD (tar over ssh) and checks the file count against the pod;
# 4. terminates the pod (retried until RunPod no longer lists it) and books elapsed x rate to CC-E1b.
set -uo pipefail
cd "$(dirname "$0")/../.."
POD=${1:?usage: detach.sh POD OUTFILE MAXMIN}; OUTF=${2:?OUTFILE}; MAXMIN=${3:?MAXMIN}
NAME=cc-e1b-h$POD; RES=character_conduct/e1b_coupling/results/h$POD
log() { echo "[detach $NAME] $(date +%H:%M:%S) $*"; }
until grep -q "\[pod\] running (detached)" "$OUTF" 2>/dev/null; do
  grep -q "Traceback\|refusing\|\[budget\]\|\[prereg\]" "$OUTF" 2>/dev/null && { log "launcher failed before the job started"; exit 1; }
  sleep 10
done
START=$(python3 -c "import os,sys; print(int(os.stat(sys.argv[1]).st_birthtime))" "$OUTF")
T=$(grep -o "\[pod\] ssh root@[0-9.]* -p [0-9]*" "$OUTF" | tail -1)
H=$(echo "$T" | sed 's/.*root@\([0-9.]*\) -p.*/\1/'); P=$(echo "$T" | sed 's/.*-p //')
L=$(grep -o "\[runpod\] pod [a-z0-9]* ([^)]*) at \$[0-9.]*/hr" "$OUTF" | tail -1)   # the booking line, not later status lines
ID=$(echo "$L" | awk '{print $3}'); RATE=$(echo "$L" | sed 's/.*at \$\([0-9.]*\)\/hr/\1/')
[ -n "$H" ] && [ -n "$P" ] && [ -n "$ID" ] && [ -n "$RATE" ] || { log "could not parse pod details from $OUTF"; exit 1; }
pkill -9 -f "launch_habit.sh $POD" ; pkill -9 -f "launch_run.py --name $NAME"
sleep 2
pgrep -f "launch_run.py --name $NAME" >/dev/null && { log "a launcher process survived; stop it by hand"; exit 1; }
log "launcher detached; pod $ID at $H:$P, \$$RATE/h, launched $(date -r "$START" +%H:%M)"
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20
     -o ServerAliveInterval=15 -i ~/.ssh/id_ed25519 -p "$P" "root@$H")
END=$(( START + (MAXMIN + 35) * 60 )); fails=0
until "${SSH[@]}" "test -f /root/job.exit" 2>/dev/null; do
  [ "$(date +%s)" -gt "$END" ] && { log "no job.exit by MAXMIN + 35 min; the self-destruct should have fired"; break; }
  "${SSH[@]}" true 2>/dev/null && fails=0 || { fails=$((fails + 1)); [ $((fails % 10)) -eq 1 ] && log "ssh failing ($fails), still waiting"; }
  sleep 60
done
log "job.exit: $("${SSH[@]}" 'cat /root/job.exit' 2>/dev/null)"
for i in 1 2 3 4 5; do
  "${SSH[@]}" "cd /root/NEEL && tar czf - $RES" 2>/dev/null | tar xzf - 2>/dev/null
  want=$("${SSH[@]}" "cd /root/NEEL && find $RES -type f | wc -l" 2>/dev/null | tr -d ' ')
  have=$(find "$RES" -type f | wc -l | tr -d ' ')
  [ -n "$want" ] && [ "$have" -ge "$want" ] && { log "pulled $have files (pod has $want)"; break; }
  log "pull attempt $i incomplete (have $have, pod ${want:-?}); retrying"; sleep 60
done
until uv run python -c "
import sys; sys.path.insert(0, 'infra'); import runpod_guard as g
if any(p['id'] == '$ID' for p in g.account().pods): g.terminate('$ID')
sys.exit(1 if any(p['id'] == '$ID' for p in g.account().pods) else 0)" 2>/dev/null; do
  log "terminate not confirmed yet; retrying"; sleep 60
done
MIN=$(( ($(date +%s) - START + 59) / 60 ))
USD=$(python3 -c "import math, sys; print(math.ceil(int(sys.argv[1]) * float(sys.argv[2]) / 60 * 1000) / 1000)" "$MIN" "$RATE")
uv run python -c "
import sys; from character_conduct.common.ledger import ledger_add
ledger_add('CC-E1b', 'gpu', float(sys.argv[1]), sys.argv[2])" "$USD" "$NAME (detached; $MIN min x \$$RATE/h)" \
  && log "terminated $ID; booked \$$USD ($MIN min x \$$RATE/h)"
