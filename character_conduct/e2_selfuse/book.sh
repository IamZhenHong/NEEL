#!/usr/bin/env bash
# Book a CC pod to character_conduct/ledger.jsonl: persona_affect/common/book_pod.sh (same arguments, same rules) with
# its ledger_add pointed at character_conduct.common.ledger.
#   bash character_conduct/e2_selfuse/book.sh LOG CC-E2 NAME MAXMIN MAXCOST
#   bash character_conduct/e2_selfuse/book.sh --bound MAXMIN MAXCOST
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)/persona_affect/common/book_pod.sh
OLD='from persona_affect.common.util import ledger_add'
NEW='from character_conduct.common.ledger import ledger_add'
grep -qF "$OLD" "$SRC" || { echo "[ledger] $SRC no longer imports ledger_add as expected; nothing booked" >&2; exit 1; }
BODY=$(sed "s/$OLD/$NEW/" "$SRC")
exec bash -c "$BODY" "$SRC" "$@"
