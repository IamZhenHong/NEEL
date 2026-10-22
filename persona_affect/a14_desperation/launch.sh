#!/usr/bin/env bash
# PA-A14: one guarded RunPod pod for one model on one jobs file (pod.sh -> a14.py run).
# Copied from persona_affect/common/launch_gen.sh @ 055d69a (plan ruling 10; the plan named e493bb7, whose cost booking
# common's final review replaced in 055d69a): its checks (integer MAXMIN >= 1; JOBS under persona_affect/ and not under
# results/logs/prep/reference), RUNPOD_ANY_CUDA=1, the HF-token extraction, the launcher started through
# common/launch_wrap.py (a TERM or HUP stops it as Ctrl-C does, so it terminates the pod) and the booking through
# common/book_pod.sh, with MINI=A14. launch_gen.sh hard-codes pod_gen.sh as the pod command, hence the copy: keep the
# two in step.
#   M=q27 JOBS=persona_affect/a14_desperation/jobs/pilot_q27.jsonl DIR=pilot_q27 MAXMIN=50 \
#     bash persona_affect/a14_desperation/launch.sh
#   # no-stock retries: LOG=persona_affect/a14_desperation/results/launch_pilot_q27 TRIES=12 WAIT=240 M=... \
#   #   bash persona/operative/retry.sh bash persona_affect/a14_desperation/launch.sh
# env, required: M       q27 | gemma
#                JOBS    jobs file, repo-relative, under persona_affect/ (only that is shipped), not under a directory
#                        called results, logs, prep or reference (the launcher's sync drops those)
#                DIR     results folder name: persona_affect/a14_desperation/results/$DIR, pulled back when the job ends;
#                        the launcher's output is tee'd to $OUT.launch.log
#                MAXMIN  whole minutes of pod time; MAXCOST = MAXMIN x $1.80/h / 60, rounded up to the cent (the guard
#                        refuses a pod whose rate x MAXMIN passes it). Size it so that the A14 ledger total + the MAXCOSTs
#                        of all A14 pods running + the judge reserve stay <= $18 (plan Task 6 Step 6)
#      optional: NAME (pa-a14-$M-$DIR)  CONTAINER_GB (120 for q27, 100 for gemma)  CLOUD (SECURE)
#                GPUS pipe-separated (NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe: an H100 would breach MAXCOST)
#                SHARD (i/n)  LIMIT_TASKS (N)  PENDING ($ of MAXCOST of other A14 pods launching or running now, 0)
# Before launching: check_cap("A14", 18.0, planned=MAXCOST + PENDING), and the jobs must all be model M's, with every
# task id in persona_affect/a14_desperation/data/tasks.jsonl (the pod reads only that file).
# When the launcher returns, common/book_pod.sh books the pod to persona_affect/ledger.jsonl (mini A14, kind gpu): the
# cost on the launcher's "[runpod] terminated <id>" line or, for a pod it never confirmed terminated, the upper bound
# MAXCOST x (MAXMIN + 30) / MAXMIN, with a WARNING that gives the command to terminate that pod. To stop a run: Ctrl-C,
# or a TERM to the process group (kill -TERM -- -<pgid>); the launcher then terminates the pod, and this books it.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
: "${M:?set M (q27 or gemma)}" "${JOBS:?set JOBS (jobs file under persona_affect/)}" "${DIR:?set DIR (results folder name)}" \
  "${MAXMIN:?set MAXMIN (whole minutes of pod time)}"
case "$M" in q27) GB=120 ;; gemma) GB=100 ;; *) echo "M must be q27 or gemma, got '$M'"; exit 2 ;; esac
case "$DIR" in ''|.|..|*/*) echo "DIR must be a plain folder name (no '/'), got '$DIR'"; exit 2 ;; esac
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
RATE=${RATE:-180}                          # cents per hour MAXCOST allows; 310 for a community H100 (owner, 10-22)
CENTS=$(( (MAXMIN * RATE + 59) / 60 ))     # rounded up to the cent
MAXCOST=$(printf '%d.%02d' $((CENTS / 100)) $((CENTS % 100)))
if [ -n "${GPUS:-}" ]; then IFS='|' read -ra GPUARR <<< "$GPUS"; else GPUARR=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); fi
uv run python -c 'import json, sys
from persona_affect.a14_desperation.tasks import load_tasks
jobs = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
models, tasks = {j["model"] for j in jobs}, load_tasks()
missing = sorted({j["task_id"] for j in jobs} - set(tasks))
if not jobs or models != {sys.argv[2]} or missing:
    raise SystemExit(f"[pre] refusing to launch: {len(jobs)} jobs, models {sorted(models)} (want {sys.argv[2]}), "
                     f"{len(missing)} task ids not in data/tasks.jsonl {missing[:3]}")' "$JOBS" "$M" \
  || { echo "[pre] refusing to launch (see above; build data/ first: python -m persona_affect.a14_desperation.tasks build)"; exit 2; }
uv run python -c 'import sys; from persona_affect.common.util import check_cap
check_cap("A14", 18.0, planned=float(sys.argv[1]) + float(sys.argv[2]))' "$MAXCOST" "$PENDING"
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
echo "[launch] $NAME: MAXMIN $MAXMIN, MAXCOST \$$MAXCOST (PENDING \$$PENDING), GPUs ${GPUARR[*]}, cloud ${CLOUD:-SECURE}"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
# A signal must not skip the booking: the script traps it and goes on to the ledger once the launcher is done; the
# launcher (through launch_wrap.py) takes INT, TERM and HUP as Ctrl-C and terminates the pod; tee ignores all three, so
# the log still gets the launcher's last lines (the terminated line)
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "${GPUARR[@]}" --container-gb "${CONTAINER_GB:-$GB}" --cloud "${CLOUD:-SECURE}" \
  --ship persona_affect --ship vigilante/pod --ship infra --results-dir "$OUT" \
  -- env M="$M" JOBS="$JOBS" DIR="$DIR" SHARD="${SHARD:-}" LIMIT_TASKS="${LIMIT_TASKS:-}" bash persona_affect/a14_desperation/pod.sh \
  2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?   # pipefail: the launcher's code ($PIPESTATUS is reset by the trap handler on bash 3.2, $? is not)
bash persona_affect/common/book_pod.sh "$LOG" "$MINI" "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
