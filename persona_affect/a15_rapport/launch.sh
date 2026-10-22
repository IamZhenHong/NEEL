#!/usr/bin/env bash
# PA-A15: one guarded RunPod pod per (phase, model), generation only (scoring and judging run locally afterwards).
# Follows persona_affect/common/launch_gen.sh: when the launcher returns, the pod's cost goes to
# persona_affect/ledger.jsonl (mini A15, kind gpu), from the last `elapsed cost ~$X` line the launcher printed.
#   M=q8 PHASE=cal MAXMIN=30 bash persona_affect/a15_rapport/launch.sh
#   # no-stock retries
#   LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh \
#     env M=q8 PHASE=cal bash persona_affect/a15_rapport/launch.sh
#   # resume a time-capped pod by shipping its partial files back (main: branches.jsonl alone)
#   D=persona_affect/a15_rapport/results/cal_q8
#   SYNC_EXTRA="$D/branches.jsonl $D/histories.json $D/history_samples.jsonl" M=q8 PHASE=cal \
#     bash persona_affect/a15_rapport/launch.sh
# env, required: M  q8 | q27 | gemma        PHASE  cal | main
#      optional: MAXMIN  whole minutes; default cal q8 30, q27 40, gemma 60; main q8 45, q27 60, gemma 90 (plan Task 4)
#                GPUS  pipe-separated; default "NVIDIA RTX A6000|NVIDIA L40S" for q8,
#                      "NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe" for q27 and gemma
#                CLOUD (SECURE)  CONTAINER_GB (q8 60, q27 120, gemma 150)  LIMIT_HIST  only the first N histories
#                SYNC_EXTRA  space-separated repo-relative files, each copied to the pod by its own --sync-extra
# NAME is pa-a15-$PHASE-$M. OUT is persona_affect/a15_rapport/results/${PHASE}_$M; the launcher pulls it back, and its
# output is tee'd to $OUT.launch.log. MAXCOST = MAXMIN x $1.80/h, or x $1.20/h when every GPU is on the 48 GB list
# (RTX A6000, L40S), rounded up to the cent; the guard refuses a pod whose rate x MAXMIN passes it.
# check_cap("A15", 7.0, planned=MAXCOST) runs first. Ships about 0.3 MB: persona_affect/{__init__.py,common,a15_rapport}
# without results, social_feedback/{__init__.py,src}, welfare_training/{__init__.py,stage0.py}, vigilante/pod, infra.
# Resume: every rerun is a new pod and results/ is never shipped, so a pod starts from what SYNC_EXTRA sends it. The
# launch refuses while OUT holds partial files that SYNC_EXTRA leaves out (a new pod would redo that work, and its pull
# would replace them), and refuses a .json file in SYNC_EXTRA that does not parse. The SYNC_EXTRA files under OUT are
# copied to $OUT.prev/ first; after the launcher returns, any whose pulled copy is missing, shorter or (a .json) does
# not parse is restored, so a pull never overwrites a longer or valid local file. A relaunch also keeps the earlier
# run_info.json, run.log and launch log in $OUT.prev/, prefixed with a UTC timestamp, before the new pod replaces them.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
MINI=A15
: "${M:?set M (q8, q27 or gemma)}" "${PHASE:?set PHASE (cal or main)}"
A100="NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe"
case "$M" in
  q8)    DEF_GPUS="NVIDIA RTX A6000|NVIDIA L40S" DEF_GB=60 ;;
  q27)   DEF_GPUS=$A100 DEF_GB=120 ;;
  gemma) DEF_GPUS=$A100 DEF_GB=150 ;;
  *) echo "M must be q8, q27 or gemma, got '$M'" >&2; exit 2 ;;
esac
case "$PHASE:$M" in
  cal:q8) DEF_MIN=30 ;;  cal:q27) DEF_MIN=40 ;;  cal:gemma) DEF_MIN=60 ;;
  main:q8) DEF_MIN=45 ;; main:q27) DEF_MIN=60 ;; main:gemma) DEF_MIN=90 ;;
  *) echo "PHASE must be cal or main, got '$PHASE'" >&2; exit 2 ;;
esac
MAXMIN=${MAXMIN:-$DEF_MIN}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'" >&2; exit 2 ;; esac
MAXMIN=$((10#$MAXMIN))
[ "$MAXMIN" -ge 1 ] || { echo "MAXMIN must be at least 1 minute" >&2; exit 2; }
if [ -n "${LIMIT_HIST:-}" ]; then
  case "$LIMIT_HIST" in
    *[!0-9]*) echo "LIMIT_HIST must be a whole number of histories, got '$LIMIT_HIST'" >&2; exit 2 ;;
  esac
  LIMIT_HIST=$((10#$LIMIT_HIST))
  [ "$LIMIT_HIST" -ge 1 ] || { echo "LIMIT_HIST must be at least 1 history" >&2; exit 2; }
fi
NAME=pa-a15-$PHASE-$M
OUT=persona_affect/a15_rapport/results/${PHASE}_$M
IFS='|' read -ra GPUARR <<< "${GPUS:-$DEF_GPUS}"
RATE=120   # cents per hour that MAXCOST allows: $1.20 when every GPU is on the 48 GB list, else $1.80 (an A100)
for g in "${GPUARR[@]}"; do
  case "$g" in "NVIDIA RTX A6000"|"NVIDIA L40S") ;; *) RATE=180 ;; esac
done
CENTS=$(( (MAXMIN * RATE + 59) / 60 ))   # rounded up to the cent
MAXCOST=$(printf '%d.%02d' $((CENTS / 100)) $((CENTS % 100)))
if [ "$PHASE" = main ]; then   # shipped with a15_rapport; pod.sh checks again on the pod
  for f in "persona_affect/a15_rapport/inputs/histories_$M.json" persona_affect/a15_rapport/inputs/config.json; do
    [ -f "$f" ] || {
      echo "$f not found: pin the cal memos and write the config (plan Task 4 Step 5) before a main pod" >&2; exit 2; }
  done
fi

# resume: the files to ship back, the partial files OUT holds, and the guard between them
json_ok() { uv run python -c 'import json, sys; json.load(open(sys.argv[1], encoding="utf-8"))' "$1" > /dev/null 2>&1; }
EXTRA=()
SYNCARGS=()
if [ -n "${SYNC_EXTRA:-}" ]; then read -ra EXTRA <<< "$SYNC_EXTRA"; fi
for i in ${EXTRA[@]+"${!EXTRA[@]}"}; do
  f=${EXTRA[$i]#./}
  case "$f" in
    /*|..|../*|*/..|*/../*) echo "SYNC_EXTRA paths must be repo-relative, without '..': $f" >&2; exit 2 ;;
  esac
  [ -f "$f" ] || { echo "SYNC_EXTRA file not found: $f" >&2; exit 2; }
  case "$f" in
    *.json) json_ok "$f" || {
      echo "SYNC_EXTRA file is not valid JSON: $f (a copy may be in $OUT.prev/; else move $OUT away)" >&2; exit 2; } ;;
  esac
  EXTRA[$i]=$f
  SYNCARGS+=(--sync-extra "$f")
done
PARTIAL=("$OUT/branches.jsonl")
if [ "$PHASE" = cal ]; then PARTIAL+=("$OUT/histories.json" "$OUT/history_samples.jsonl"); fi
PRESENT="" UNSHIPPED=""
for f in "${PARTIAL[@]}"; do
  [ -f "$f" ] || continue
  PRESENT="${PRESENT:+$PRESENT }$f"
  hit=""
  for e in ${EXTRA[@]+"${EXTRA[@]}"}; do if [ "$e" = "$f" ]; then hit=1; fi; done
  [ -n "$hit" ] || UNSHIPPED="$UNSHIPPED $f"
done
if [ -n "$UNSHIPPED" ]; then
  echo "$OUT holds partial results that SYNC_EXTRA does not ship:$UNSHIPPED" >&2
  echo "A new pod would redo that work, and its pull would replace them. To resume, run again with" >&2
  echo "  SYNC_EXTRA=\"$PRESENT\"" >&2
  echo "To start over, move $OUT away first." >&2
  exit 2
fi
uv run python -c 'import sys; from persona_affect.common.util import check_cap
check_cap("A15", 7.0, planned=float(sys.argv[1]))' "$MAXCOST"
SAVED=()
for f in ${EXTRA[@]+"${EXTRA[@]}"}; do
  case "$f" in "$OUT"/*) ;; *) continue ;; esac   # the pull rewrites only OUT
  rel=${f#"$OUT"/}
  mkdir -p "$(dirname "$OUT.prev/$rel")"
  cp -p "$f" "$OUT.prev/$rel"
  SAVED+=("$rel")
done
if [ ${#EXTRA[@]} -gt 0 ]; then
  echo "[resume] shipping ${#EXTRA[@]} file(s) with --sync-extra; ${#SAVED[@]} under $OUT copied to $OUT.prev/"
fi
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
keep_old() {   # keep_old FILE NAME: a relaunch keeps the earlier pod's FILE as $OUT.prev/$STAMP.NAME
  if [ -f "$1" ]; then
    mkdir -p "$OUT.prev" && cp -p "$1" "$OUT.prev/$STAMP.$2" && echo "[relaunch] kept $1 as $OUT.prev/$STAMP.$2"
  fi
}
keep_old "$OUT/run_info.json" run_info.json
keep_old "$OUT/run.log" run.log
keep_old "$OUT.launch.log" launch.log

HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
PODENV=(M="$M" PHASE="$PHASE" OUT="$OUT")
if [ -n "${LIMIT_HIST:-}" ]; then PODENV+=(LIMIT_HIST="$LIMIT_HIST"); fi
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
# Ctrl-C (or a kill) must not skip the booking: trap the signals here so the script goes on to the ledger, and keep tee
# alive through INT, because the launcher's cleanup (terminate the pod, print the final cost line) comes after the signal
trap 'echo "[launch] interrupted: booking what the launcher log shows"' INT TERM HUP
uv run python infra/launch_run.py --name "$NAME" --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "${GPUARR[@]}" --container-gb "${CONTAINER_GB:-$DEF_GB}" --cloud "${CLOUD:-SECURE}" \
  --ship persona_affect/__init__.py --ship persona_affect/common --ship persona_affect/a15_rapport \
  --ship social_feedback/__init__.py --ship social_feedback/src --ship welfare_training/__init__.py \
  --ship welfare_training/stage0.py --ship vigilante/pod --ship infra --exclude persona_affect/a15_rapport/results \
  ${SYNCARGS[@]+"${SYNCARGS[@]}"} --results-dir "$OUT" \
  -- env "${PODENV[@]}" bash persona_affect/a15_rapport/pod.sh \
  2>&1 | (trap '' INT; exec tee "$LOG")
rc=$?   # pipefail: the launcher's code ($PIPESTATUS is reset by the trap handler on bash 3.2, $? is not)
trap - INT TERM HUP
set -e

# Book the pod: the last cost line the launcher printed ("[runpod] terminated ...; elapsed cost ~$X" comes after the
# "[pod] job exit code ..." one). A pod that was created but left no cost line (launcher killed) is booked at MAXCOST, an
# upper bound. If no pod was ever created (no stock, a refusal) nothing was spent and nothing is booked, so retry.sh rounds
# cost the ledger nothing.
COST=$(grep -o 'elapsed cost ~\$[0-9][0-9.]*' "$LOG" | tail -1 | sed 's/.*~\$//' || true)
USD="" NOTE=$NAME
if [ -n "$COST" ]; then USD=$COST
elif grep -q '\[runpod\] pod [^ ]* (' "$LOG"; then USD=$MAXCOST NOTE="$NAME (upper bound, no cost line)"
fi
if [ -z "$USD" ]; then
  echo "[ledger] no pod was created (no cost line and no pod line in $LOG): nothing booked"
else
  uv run python -c 'import sys; from persona_affect.common.util import ledger_add; ledger_add(sys.argv[1], "gpu", float(sys.argv[2]), note=sys.argv[3])' "$MINI" "$USD" "$NOTE" \
    && echo "[ledger] $MINI gpu \$$USD ($NOTE)" \
    || echo "[ledger] FAILED to book \$$USD for $MINI ($NOTE): add it with persona_affect.common.util.ledger_add"
fi

# resume: a pulled copy that is missing, shorter than the file shipped from here or (a .json) unparsable is replaced by
# the local copy
for rel in ${SAVED[@]+"${SAVED[@]}"}; do
  cur="$OUT/$rel" prev="$OUT.prev/$rel" why=""
  if [ ! -f "$cur" ]; then why="missing"
  elif [ "$(wc -c < "$cur")" -lt "$(wc -c < "$prev")" ]; then why="shorter"
  else
    case "$cur" in *.json) json_ok "$cur" || why="not valid JSON" ;; esac
  fi
  if [ -n "$why" ]; then
    cp -p "$prev" "$cur" && echo "[resume] restored $cur: the pulled copy was $why" \
      || echo "[resume] FAILED to restore $cur: copy $prev back by hand"
  fi
done
exit "$rc"
