#!/usr/bin/env bash
# Rerun a guarded launcher until it gets a GPU (each failed round set costs nothing).
#   LOG=persona/history/launch_x TRIES=12 WAIT=240 bash persona/history/retry.sh <launcher command...>
# Stops at the first attempt that did not end in "no GPU stock" (success or a real error).
set -u
for i in $(seq 1 ${TRIES:-12}); do
  "$@" > ${LOG:?}.try$i.log 2>&1; rc=$?
  grep -q "no GPU stock after all rounds" ${LOG}.try$i.log || exit $rc
  echo "[retry] try $i: no stock; waiting ${WAIT:-240}s" >> ${LOG}.retry.log
  sleep ${WAIT:-240}
done
echo "[retry] gave up after ${TRIES:-12} tries" >> ${LOG}.retry.log
exit 1
