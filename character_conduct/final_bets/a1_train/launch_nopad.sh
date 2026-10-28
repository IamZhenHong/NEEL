#!/usr/bin/env bash
# CC-F-A1 deviation 10 (Option A): train one no-padding control (V2n / V3n / V4n) on a serving replica whose ODCV shard
# has emptied. Ships train_a1.py and the nopad data, then runs train_on_replica.sh detached; the replica's server
# restarts with the new adapter (named V2n etc.) when training ends.
#   bash character_conduct/final_bets/a1_train/launch_nopad.sh HOST PORT V2n
set -euo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
HOST=${1:?}; PORT=${2:?}; V=${3:?}
SHA=ea635874aca7a695853925ae5b1543143e2415f3120b5e35f0df3df600d46edd
O=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -i ~/.ssh/id_ed25519)
scp "${O[@]}" -P "$PORT" "$H/train_a1.py" "root@$HOST:/root/NEEL/$H/train_a1.py"
scp "${O[@]}" -P "$PORT" "$H/data/train_versions_nopad.jsonl.gz" "root@$HOST:/root/NEEL/$H/data/"
# the same 1,000 steps as every other arm (pod S's timing smoke), not a fresh smoke on this replica
ssh "${O[@]}" -p "$PORT" "root@$HOST" "mkdir -p /root/NEEL/$H/results/train; [ -f /root/NEEL/$H/results/train/steps.json ] || echo '{\"steps\": 1000, \"rule\": \"pod S timing smoke (A100 SXM, 2.593 s/step), deviation 8\"}' > /root/NEEL/$H/results/train/steps.json; cat /root/NEEL/$H/results/train/steps.json"
ssh "${O[@]}" -p "$PORT" "root@$HOST" "cd /root/NEEL && DATA=$H/data/train_versions_nopad.jsonl.gz setsid nohup bash $H/train_on_replica.sh '$V:1' $SHA > /root/nopad_$V.log 2>&1 < /dev/null & sleep 2; echo started; ls $H/results/train/steps.json 2>&1"
