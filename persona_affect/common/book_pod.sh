#!/usr/bin/env bash
# Book one RunPod pod's GPU cost to persona_affect/ledger.jsonl (kind gpu) from its launcher log, once per launch.
# launch_gen.sh calls it when the launcher returns; a mini's own launch script should call it too, so that a fix to the
# booking lands everywhere:
#   bash persona_affect/common/book_pod.sh LOG MINI NAME MAXMIN MAXCOST    # from any directory; LOG is read from there
#   bash persona_affect/common/book_pod.sh --bound MAXMIN MAXCOST          # print the upper bound below, book nothing
# It reads the pod line "[runpod] pod <id> (...)" and the line infra/runpod_guard.py prints once it has terminated the
# pod, "[runpod] terminated <id>; elapsed cost ~$X", and books:
#   - a terminated line for the pod: $X, note NAME. Only that line is final. The launcher also prints "elapsed cost"
#     when the job exits, before it pulls the results and terminates the pod; a launcher killed in between leaves a pod
#     running that this earlier figure would under-book.
#   - a pod line but no terminated line for it: the pod may still be running, until its boot self-destruct MAXMIN + 30 min
#     after it started, at no more than MAXCOST x 60 / MAXMIN per hour (the guard refuses a dearer pod). So it books that
#     upper bound, MAXCOST x (MAXMIN + 30) / MAXMIN rounded up to the cent (the image pull comes on top), with the note
#     "NAME (upper bound, pod not confirmed terminated)", and prints a WARNING with the pod id and the command that
#     terminates it (runpod_guard.py --kill-ours only kills provocation-* pods).
#   - no pod line: no pod was created (no stock, a refusal), nothing was spent and nothing is booked, so retry.sh rounds
#     cost the ledger nothing.
# Exits 0 when the pod was booked or there was nothing to book, 1 when the log is missing or the booking failed, 2 on bad
# arguments. TERM and HUP are ignored: the stop signal that ended the launcher must not cut the booking short.
set -uo pipefail
trap '' TERM HUP
ROOT=$(cd "$(dirname "$0")/../.." && pwd)

usage() { echo "usage: book_pod.sh LOG MINI NAME MAXMIN MAXCOST, or book_pod.sh --bound MAXMIN MAXCOST" >&2; exit 2; }
check_limits() {   # MAXMIN MAXCOST
  case "$1" in ''|*[!0-9]*) echo "book_pod.sh: MAXMIN must be a whole number of minutes, got '$1'" >&2; return 1;; esac
  [ "$((10#$1))" -gt 0 ] || { echo "book_pod.sh: MAXMIN must be at least 1 minute, got '$1'" >&2; return 1; }
  [[ "$2" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "book_pod.sh: MAXCOST must be a number of USD, got '$2'" >&2; return 1; }
}
bound() {          # MAXMIN MAXCOST -> MAXCOST x (MAXMIN + 30) / MAXMIN, rounded up to the cent
  python3 -c 'import sys; from decimal import Decimal, ROUND_CEILING; m, c = int(sys.argv[1]), Decimal(sys.argv[2]); print((c * (m + 30) / m).quantize(Decimal("0.01"), rounding=ROUND_CEILING))' "$((10#$1))" "$2"
}

if [ "${1:-}" = "--bound" ]; then
  [ $# -eq 3 ] || usage
  check_limits "$2" "$3" || exit 2
  bound "$2" "$3"
  exit
fi
[ $# -eq 5 ] || usage
LOG=$1 MINI=$2 NAME=$3 MAXMIN=$4 MAXCOST=$5
check_limits "$MAXMIN" "$MAXCOST" || exit 2
[ -n "$MINI" ] || { echo "book_pod.sh: MINI (the ledger id) is empty" >&2; exit 2; }
case "$LOG" in /*) ;; *) LOG="$PWD/$LOG";; esac

if [ ! -f "$LOG" ]; then
  echo "[ledger] WARNING: no launcher log at $LOG, so it cannot tell whether a pod was created; nothing booked. Look for" \
       "a $NAME pod in the RunPod console." >&2
  exit 1
fi
POD=$(sed -n 's/.*\[runpod\] pod \([^ ]*\) (.*/\1/p' "$LOG" | tail -1)
[ -n "$POD" ] || POD=$(sed -n 's/.*\[runpod\] terminated \([^;]*\); elapsed cost.*/\1/p' "$LOG" | tail -1)
if [ -z "$POD" ]; then
  echo "[ledger] no pod was created (no pod line in $LOG): nothing booked"
  exit 0
fi
COST=$(grep -F "[runpod] terminated $POD; elapsed cost ~\$" "$LOG" | tail -1 | sed 's/.*elapsed cost ~\$\([0-9][0-9.]*\).*/\1/')
if [[ "$COST" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  USD=$COST NOTE=$NAME
else
  USD=$(bound "$MAXMIN" "$MAXCOST") NOTE="$NAME (upper bound, pod not confirmed terminated)"
  {
    echo "[ledger] WARNING: pod $POD was created, but the launcher never confirmed that it terminated it (no"
    echo "  \"[runpod] terminated $POD\" line in $LOG). It may still be running, and billing, until its boot self-destruct"
    echo "  $((10#$MAXMIN + 30)) min after it started. Look for it in the RunPod console, or terminate it now, from the repo root:"
    echo "    uv run python -c \"import sys; sys.path.insert(0, 'infra'); from runpod_guard import terminate; terminate('$POD')\""
    echo "  (runpod_guard.py --kill-ours only kills provocation-* pods.) Booking the upper bound \$$USD = MAXCOST \$$MAXCOST"
    echo "  x ($((10#$MAXMIN)) + 30) / $((10#$MAXMIN))."
  } >&2
fi
cd "$ROOT" || exit 1
if uv run python -c 'import sys; from persona_affect.common.util import ledger_add; ledger_add(sys.argv[1], "gpu", float(sys.argv[2]), note=sys.argv[3])' "$MINI" "$USD" "$NOTE"; then
  echo "[ledger] $MINI gpu \$$USD ($NOTE)"
else
  echo "[ledger] FAILED to book \$$USD for $MINI ($NOTE): add it with persona_affect.common.util.ledger_add" >&2
  exit 1
fi
