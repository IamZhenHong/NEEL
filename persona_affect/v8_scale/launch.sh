#!/usr/bin/env bash
# PA-V8: one guarded RunPod pod per (stage, group), as persona/history/launch_e.sh: pod.sh runs every model of the group in
# turn, one vLLM process each. When the launcher returns, `v8 ledger` books the pod's cost to persona_affect/ledger.jsonl
# (mini V8, kind gpu) from the launcher output, as persona_affect/common/launch_gen.sh does.
#   STAGE=calib GROUP=small MAXMIN=60 PENDING=1.80 bash persona_affect/v8_scale/launch.sh
#   # no-stock retries (each try is one run of this script: one log, one booking); retry.sh keeps its own copies in
#   # ${LOG}.try<i>.log, so its folder must exist
#   mkdir -p persona_affect/v8_scale/results && STAGE=calib GROUP=big MAXMIN=60 PENDING=0.90 \
#     LOG=persona_affect/v8_scale/results/calib_big TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona_affect/v8_scale/launch.sh
# env, required: STAGE  calib | main | rewrite (main and rewrite need persona_affect/v8_scale/frozen.json)
#                GROUP  small: "NVIDIA A40" or "NVIDIA RTX A6000", MAXCOST at $0.90/h, 80 GB container disk
#                       big:   "NVIDIA A100-SXM4-80GB" or "NVIDIA A100 80GB PCIe", $1.80/h, 120 GB
#                MAXMIN whole minutes of pod time; MAXCOST = MAXMIN x the group's $/h / 60, rounded up to the cent (the
#                       guard refuses a pod whose rate x MAXMIN passes it)
#      optional: NAME (v8-$STAGE-$GROUP)  CLOUD (SECURE)  PENDING (0; the MAXCOST of the other V8 pod launching now)
# check_cap("V8", 6.0, planned=MAXCOST + PENDING) runs first. HF_TOKEN comes from .env. Ships about 1.6 MB. The pod's
# results/${STAGE}_${GROUP} is pulled back to persona_affect/v8_scale/results/${STAGE}_${GROUP}; the launcher's output is
# tee'd to results/${STAGE}_${GROUP}.launch.log, overwritten by each try, and booked once per try.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
: "${STAGE:?set STAGE (calib, main or rewrite)}" "${GROUP:?set GROUP (small or big)}" "${MAXMIN:?set MAXMIN (whole minutes)}"
case "$STAGE" in calib|main|rewrite) ;; *) echo "STAGE must be calib, main or rewrite, got '$STAGE'" >&2; exit 2 ;; esac
case "$GROUP" in
  small) GPUS=("NVIDIA A40" "NVIDIA RTX A6000") RATE=90 GB=120 ;;                       # cents per hour, container GB
  big)   GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe") RATE=180 GB=200 ;;   # 120 ran out on 10-22 (xet cache)
  *) echo "GROUP must be small or big, got '$GROUP'" >&2; exit 2 ;;
esac
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'" >&2; exit 2 ;; esac
MAXMIN=$((10#$MAXMIN))
[ "$MAXMIN" -ge 1 ] || { echo "MAXMIN must be at least 1 minute" >&2; exit 2; }
CENTS=$(( (MAXMIN * RATE + 59) / 60 ))   # rounded up to the cent
MAXCOST=$(printf '%d.%02d' $((CENTS / 100)) $((CENTS % 100)))
PENDING=${PENDING:-0}
NAME=${NAME:-v8-$STAGE-$GROUP}
OUT=persona_affect/v8_scale/results/${STAGE}_${GROUP}
LOG=$OUT.launch.log
uv run python -c 'import sys; from persona_affect.common.util import check_cap
check_cap("V8", 6.0, planned=float(sys.argv[1]) + float(sys.argv[2]))' "$MAXCOST" "$PENDING"
if [ "$STAGE" != calib ] && [ ! -f persona_affect/v8_scale/frozen.json ]; then
  echo "persona_affect/v8_scale/frozen.json not found: $STAGE jobs are built only after the freeze (v8 calib-report)" >&2
  exit 2
fi
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
mkdir -p "$(dirname "$OUT")"
echo "[launch] $NAME: MAXMIN $MAXMIN, MAXCOST \$$MAXCOST (planned with PENDING \$$PENDING), GPUs ${GPUS[*]}, cloud ${CLOUD:-SECURE}"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
# Ctrl-C (or a kill) must not skip the booking: trap the signals here so the script goes on to the ledger, and keep tee
# alive through INT, because the launcher's cleanup (terminate the pod, print the final cost line) comes after the signal
trap 'echo "[launch] interrupted: booking what the launcher log shows"' INT TERM HUP
uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "${GPUS[@]}" --container-gb "$GB" --cloud "${CLOUD:-SECURE}" \
  --ship persona_affect --ship persona/exemplar/tasks.py --ship persona/exemplar/mini0.py \
  --ship persona/history/runner.py --ship vigilante/src --ship vigilante/__init__.py --ship vigilante/pod --ship infra \
  --results-dir "$OUT" -- env STAGE="$STAGE" GROUP="$GROUP" MODELS="${MODELS:-}" bash persona_affect/v8_scale/pod.sh \
  2>&1 | (trap '' INT; exec tee "$LOG")
rc=$?   # pipefail: the launcher's code ($PIPESTATUS is reset by the trap handler on bash 3.2, $? is not)
trap - INT TERM HUP
set -e

# Book this try once, from its own log: the last `elapsed cost ~$X` line; MAXCOST (an upper bound) when a pod was
# created but left no cost line; nothing when no pod was created (no stock, a refusal).
uv run python -m persona_affect.v8_scale.v8 ledger --log "$LOG" --name "$NAME" --maxcost "$MAXCOST" \
  || echo "[ledger] FAILED to book $NAME for V8: run v8 ledger --log $LOG --name $NAME --maxcost $MAXCOST by hand"
exit "$rc"
