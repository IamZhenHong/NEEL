#!/usr/bin/env bash
# split_after_wave0.sh RUN "EXTRA_FLAGS" "ARMS_MAIN" LP_MAIN "ARMS_S2" LP_S2 ["ARMS_S3" LP_S3]
# Waits for RUN's "wave 0 done", stops it, then runs waves 1-5 as subsets: RUN (main arms, main port) and RUNs2/RUNs3
# (--start-wave 1 --notes-from RUN) on their own pods. Same per-arm procedure; only the batching across arms changes.
set -u
RUN=$1; FL=$2; A1=$3; L1=$4; A2=$5; L2=$6; A3=${7:-}; L3=${8:-}
cd "$(dirname "$0")/../.."
until grep -q "wave 0 done" "state_attack/logs/$RUN.log"; do sleep 10; done
pkill -f "attack --run $RUN " ; sleep 3
start() {  # name arms port extra
  SA_URL=http://127.0.0.1:$3 nohup uv run --with anthropic python -u -m state_attack.src.attack --run "$1" $FL --arms $2 $4 \
      >> "state_attack/logs/$1.log" 2>&1 &
  echo "[split] $1 on :$3 arms: $2 $(date +%T)"
}
start "$RUN" "$A1" "$L1" ""
start "${RUN}s2" "$A2" "$L2" "--start-wave 1 --notes-from $RUN"
[ -n "$A3" ] && start "${RUN}s3" "$A3" "$L3" "--start-wave 1 --notes-from $RUN"
