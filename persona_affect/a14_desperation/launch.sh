#!/usr/bin/env bash
# PA-A14: one guarded RunPod pod for one model on one jobs file (pod.sh -> a14.py run). When the launcher returns, the
# pod's cost goes to persona_affect/ledger.jsonl (mini A14, kind gpu): the last `elapsed cost ~$X` line it printed.
# Copied from persona_affect/common/launch_gen.sh @ e493bb7 (plan ruling 10): its checks (integer MAXMIN; JOBS under
# persona_affect/ and not under results/logs/prep/reference), RUNPOD_ANY_CUDA=1, the HF-token extraction and the
# cost-booking block, with MINI=A14. launch_gen.sh hard-codes pod_gen.sh as the pod command, hence the copy: keep the
# two in step.
#   M=q27 JOBS=persona_affect/a14_desperation/jobs/pilot_q27.jsonl DIR=pilot_q27 MAXMIN=50 \
#     bash persona_affect/a14_desperation/launch.sh
#   # no-stock retries: LOG=persona_affect/a14_desperation/results/launch_pilot_q27 TRIES=12 WAIT=240 M=... \
#   #   bash persona/operative/retry.sh bash persona_affect/a14_desperation/launch.sh
# env, required: M       q27 | gemma
#                JOBS    jobs file, repo-relative, under persona_affect/ (only that is shipped), not under a directory
#                        called results, logs, prep or reference (the launcher's sync drops those)
#                DIR     results folder: persona_affect/a14_desperation/results/$DIR, pulled back when the job ends; the
#                        launcher's output is tee'd to $OUT.launch.log
#                MAXMIN  whole minutes of pod time; MAXCOST = MAXMIN x $1.80/h / 60, rounded up to the cent (the guard
#                        refuses a pod whose rate x MAXMIN passes it). Size it so that the A14 ledger total + the MAXCOSTs
#                        of all A14 pods running + the judge reserve stay <= $18 (plan Task 6 Step 6)
#      optional: NAME (pa-a14-$M-$DIR)  CONTAINER_GB (120 for q27, 100 for gemma)  CLOUD (SECURE)
#                GPUS pipe-separated (NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe: an H100 would breach MAXCOST)
#                SHARD (i/n)  LIMIT_TASKS (N)  PENDING ($ of MAXCOST of other A14 pods launching or running now, 0)
# check_cap("A14", 18.0, planned=MAXCOST + PENDING) runs first and stops the launch if the ledger would pass $18.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
: "${M:?set M (q27 or gemma)}" "${JOBS:?set JOBS (jobs file under persona_affect/)}" "${DIR:?set DIR (results folder name)}" \
  "${MAXMIN:?set MAXMIN (whole minutes of pod time)}"
case "$M" in q27) GB=120 ;; gemma) GB=100 ;; *) echo "M must be q27 or gemma, got '$M'"; exit 2 ;; esac
MINI=A14
NAME=${NAME:-pa-a14-$M-$DIR} PENDING=${PENDING:-0}
JOBS=${JOBS#./}
OUT=persona_affect/a14_desperation/results/$DIR
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXMIN=$((10#$MAXMIN))
[ "$MAXMIN" -ge 1 ] || { echo "MAXMIN must be at least 1 minute"; exit 2; }
case "$JOBS" in persona_affect/*) ;; *) echo "JOBS must be a repo-relative path under persona_affect/ (only that is shipped): $JOBS"; exit 2;; esac
case "/$JOBS/" in */results/*|*/logs/*|*/prep/*|*/reference/*)
  echo "JOBS sits under a results/, logs/, prep/ or reference/ directory, which the launcher's sync drops: $JOBS"; exit 2;; esac
[ -f "$JOBS" ] || { echo "JOBS not found: $JOBS"; exit 2; }
CENTS=$(( (MAXMIN * 180 + 59) / 60 ))      # $1.80/h, rounded up to the cent
MAXCOST=$(printf '%d.%02d' $((CENTS / 100)) $((CENTS % 100)))
if [ -n "${GPUS:-}" ]; then IFS='|' read -ra GPUARR <<< "$GPUS"; else GPUARR=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); fi
uv run python -c 'import sys; from persona_affect.common.util import check_cap
check_cap("A14", 18.0, planned=float(sys.argv[1]) + float(sys.argv[2]))' "$MAXCOST" "$PENDING"
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
echo "[launch] $NAME: MAXMIN $MAXMIN, MAXCOST \$$MAXCOST (PENDING \$$PENDING), GPUs ${GPUARR[*]}, cloud ${CLOUD:-SECURE}"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
# Ctrl-C (or a kill) must not skip the booking: trap the signals here so the script goes on to the ledger, and keep tee
# alive through INT, because the launcher's cleanup (terminate the pod, print the final cost line) comes after the signal
trap 'echo "[launch] interrupted: booking what the launcher log shows"' INT TERM HUP
uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "${GPUARR[@]}" --container-gb "${CONTAINER_GB:-$GB}" --cloud "${CLOUD:-SECURE}" \
  --ship persona_affect --ship vigilante/pod --ship infra --results-dir "$OUT" \
  -- env M="$M" JOBS="$JOBS" DIR="$DIR" SHARD="${SHARD:-}" LIMIT_TASKS="${LIMIT_TASKS:-}" bash persona_affect/a14_desperation/pod.sh \
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
exit "$rc"
