#!/usr/bin/env bash
# Book one RunPod pod's GPU cost to self_concept/ledger.jsonl: persona_affect/common/book_pod.sh (same arguments, same
# rules) with its ledger_add import pointed at self_concept.common.ledger, so a fix to the shared booking lands here too.
#   bash self_concept/common/book_pod.sh LOG MINI NAME MAXMIN MAXCOST
#   bash self_concept/common/book_pod.sh --bound MAXMIN MAXCOST
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)/persona_affect/common/book_pod.sh
OLD='from persona_affect.common.util import ledger_add'
NEW='import self_concept.common.ledger; from self_concept.common.ledger import ledger_add'
grep -qF "$OLD" "$SRC" || { echo "[ledger] $SRC no longer imports ledger_add as expected; nothing booked" >&2; exit 1; }
BODY=$(sed "s/$OLD/$NEW/" "$SRC")
exec bash -c "$BODY" "$SRC" "$@"   # $0 = the original path, so its ROOT lookup still works
