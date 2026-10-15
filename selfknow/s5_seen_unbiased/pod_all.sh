#!/usr/bin/env bash
# s5 on ONE launcher-managed pod (saves a second ~20 min pod start-up): pilot -> hold for the freeze -> main.
#   1. Pilot (gate) on Qwen3-32B -> results/pilot/. Crash -> stop. Gate fail -> Gemma 4 31B pilot (card fallback,
#      tag _gemma) and stop (a Gemma main run would need its own tested path and amendment).
#   2. Gate pass -> hold, printing a heartbeat, until the file selfknow/s5_seen_unbiased/GO appears (written over ssh
#      only AFTER the s5-freeze commit and tag) or HOLD_MIN minutes pass (then stop). GO holds "<cue> <expl-deadline>".
#   3. Main stage with that cue -> results/main/. The code is the one synced at launch (== the frozen commit's code;
#      README records the sha256 of every code file at launch and at freeze).
#   bash selfknow/s5_seen_unbiased/launch.sh s5-2 105 3.15 "" selfknow/s5_seen_unbiased/pod_all.sh
# Local flow test with a stub instead of the model: S5_PY=<stub> HOLD_MIN=1 bash selfknow/s5_seen_unbiased/pod_all.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
D=selfknow/s5_seen_unbiased
RP=$D/results/pilot; RM=$D/results/main
PY=${S5_PY:-python}
HOLD_MIN=${HOLD_MIN:-35}
mkdir -p $RP $RM
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a $RP/pod.log
$PY -m selfknow.s5_seen_unbiased.pod_s5 --stage pilot --out $RP 2>&1 \
    | grep --line-buffered -v "Loading weights\|Fetching" | tee -a $RP/pod.log
G=$RP/pilot_gate.json
if [ ! -f $G ]; then echo "[s5] pilot produced no gate file; stopping" | tee -a $RP/pod.log; exit 1; fi
if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$G'))['pass'] else 1)"; then
    echo "[s5] Qwen3-32B gate FAILED; Gemma 4 31B fallback pilot" | tee -a $RP/pod.log
    rm -rf /root/hf/hub/models--Qwen--Qwen3-32B 2>/dev/null
    $PY -m selfknow.s5_seen_unbiased.pod_s5 --stage pilot --model google/gemma-4-31B-it --tag _gemma --out $RP 2>&1 \
        | grep --line-buffered -v "Loading weights\|Fetching" | tee -a $RP/pod.log
    echo "[s5] stopping after the fallback pilot" | tee -a $RP/pod.log
    exit 0
fi
echo "[s5] gate passed; holding up to $HOLD_MIN min for $D/GO" | tee -a $RP/pod.log
t=0
while [ ! -f $D/GO ] && [ $t -lt $((HOLD_MIN * 60)) ]; do
    sleep 30; t=$((t + 30)); echo "[s5] holding ${t}s"
done
if [ ! -f $D/GO ]; then echo "[s5] no GO after $HOLD_MIN min; stopping" | tee -a $RP/pod.log; exit 0; fi
read CUE DEADLINE < $D/GO
echo "[s5] GO: cue $CUE, exploratory deadline $DEADLINE min" | tee -a $RM/pod.log
$PY -m selfknow.s5_seen_unbiased.pod_s5 --stage main --cue "$CUE" --expl-deadline-min "$DEADLINE" --out $RM 2>&1 \
    | grep --line-buffered -v "Loading weights\|Fetching" | tee -a $RM/pod.log
echo "[s5] main exit ${PIPESTATUS[0]}" | tee -a $RM/pod.log
