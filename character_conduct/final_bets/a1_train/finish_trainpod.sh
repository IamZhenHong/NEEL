#!/usr/bin/env bash
# CC-F-A1 (deviation 8b): finish a training-only pod. Waits until every NAME's adapter is in its /root/adapters_in
# (DONE marker + "-> NAME" line in trainseq.log), uploads each to HF (hf_upload.sh), pulls each onto pod S
# (hf_download.sh), pulls the pod's training records (no weights) into results/pod_<POD>/, terminates the pod
# (retried until RunPod no longer lists it) and books elapsed x rate to CC-F-A1. With KEEP=1 it stops after the
# uploads and downloads (the pod keeps running).
#   bash character_conduct/final_bets/a1_train/finish_trainpod.sh POD HOST PORT POD_ID RATE START_EPOCH NAME [NAME ...]
set -uo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
POD=${1:?}; HOST=${2:?}; PORT=${3:?}; ID=${4:?}; RATE=${5:?}; START=${6:?}; shift 6
S_HOST=64.247.196.114; S_PORT=16082
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -i ~/.ssh/id_ed25519 -p "$PORT" "root@$HOST")
log() { echo "[finish $POD] $(date +%H:%M:%S) $*"; }
for N in "$@"; do
  until "${SSH[@]}" "grep -q -- '-> $N ' /root/NEEL/$H/results/train/trainseq.log 2>/dev/null"; do
    "${SSH[@]}" "grep -l 'Traceback\|out of memory' /root/NEEL/$H/results/train/train_*.log 2>/dev/null" | grep -q . && { log "training error"; "${SSH[@]}" "tail -5 /root/train.log"; exit 1; }
    sleep 60
  done
  log "$N saved; uploading"
  bash $H/hf_upload.sh "$HOST" "$PORT" "$N" 2>&1 | grep "\[hf\]"
  bash $H/hf_download.sh "$S_HOST" "$S_PORT" "$N" 2>&1 | grep "\[hf-dl\]\|Error\|assert"
done
mkdir -p "$H/results/pod_$POD"
"${SSH[@]}" "cd /root/NEEL/$H/results && tar cf - train/*.json train/*.log train/steps.json serve/pod.log 2>/dev/null" | tar xf - -C "$H/results/pod_$POD"
"${SSH[@]}" "cp /root/train.log /root/NEEL/$H/results/train/podtrain.log 2>/dev/null; cat /root/train.log" > "$H/results/pod_$POD/train_outer.log" 2>/dev/null
log "records pulled: $(find "$H/results/pod_$POD" -type f | wc -l | tr -d ' ') files"
[ "${KEEP:-}" = 1 ] && { log "KEEP=1: pod left running"; exit 0; }
until uv run python -c "
import sys; sys.path.insert(0, 'infra'); import runpod_guard as g
if any(p['id'] == '$ID' for p in g.account().pods): g.terminate('$ID')
sys.exit(1 if any(p['id'] == '$ID' for p in g.account().pods) else 0)" 2>/dev/null; do log "terminate not confirmed; retrying"; sleep 30; done
MIN=$(( ($(date +%s) - START + 59) / 60 ))
USD=$(python3 -c "import math, sys; print(math.ceil(int(sys.argv[1]) * float(sys.argv[2]) / 60 * 1000) / 1000)" "$MIN" "$RATE")
uv run python -c "
import sys; from character_conduct.common.ledger import ledger_add
ledger_add('CC-F-A1', 'gpu', float(sys.argv[1]), sys.argv[2])" "$USD" "cc-f-a1-$(echo "$POD" | tr 'A-Z' 'a-z') $ID (training pod, detached; $MIN min x \$$RATE/h)"
log "terminated $ID; booked \$$USD ($MIN min x \$$RATE/h)"
