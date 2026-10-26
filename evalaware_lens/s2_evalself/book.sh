#!/usr/bin/env bash
# Book the shared EL-S2 / EL-C3 pod to evalaware_lens/ledger.jsonl: persona_affect/common/book_pod.sh (same arguments,
# same rules) with its ledger_add pointed at evalaware_lens.s2_evalself.book (the pre-registered 55 / 45 split).
#   bash evalaware_lens/s2_evalself/book.sh LOG EL-S2+EL-C3 NAME MAXMIN MAXCOST
#   bash evalaware_lens/s2_evalself/book.sh --bound MAXMIN MAXCOST
set -uo pipefail
SRC=$(cd "$(dirname "$0")/../.." && pwd)/persona_affect/common/book_pod.sh
OLD='from persona_affect.common.util import ledger_add'
NEW='from evalaware_lens.s2_evalself.book import ledger_add'
grep -qF "$OLD" "$SRC" || { echo "[ledger] $SRC no longer imports ledger_add as expected; nothing booked" >&2; exit 1; }
BODY=$(sed "s/$OLD/$NEW/" "$SRC")
exec bash -c "$BODY" "$SRC" "$@"
