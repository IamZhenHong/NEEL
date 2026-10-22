#!/usr/bin/env bash
# PA-A8: one guarded RunPod pod per model, generation only (scoring and judging run locally afterwards). Mirrors
# persona/inference/launch_distances.sh. When the launcher returns, the pod's cost goes to persona_affect/ledger.jsonl (mini A8,
# kind gpu), as persona_affect/common/launch_gen.sh does: the last `elapsed cost ~$X` line the launcher printed.
#   M=q27 bash persona_affect/a8_rejection/launch.sh                  # g3 | gemma | q27 | l8
#   # smoke run: the first 24 branches of l8, into a directory of its own
#   M=l8 LIMIT=24 MAXMIN=30 OUT=persona_affect/a8_rejection/results/smoke_l8 NAME=pa-a8-smoke-l8 bash persona_affect/a8_rejection/launch.sh
#   # no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona_affect/a8_rejection/launch.sh
# env, required: M  g3 | gemma | q27 | l8
#      optional: GPUS  pipe-separated, because the names have spaces; default "NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe" for g3,
#                      gemma and q27, "NVIDIA A40|NVIDIA L40S" for l8
#                MAXMIN (100; 150 for gemma: 31B weights leave little KV room and its branch turns have run ~0.36 items/s)
#                MAXCOST (MAXMIN x $1.80/h, so 3.00 and, for gemma, 4.50; the guard refuses a pod whose rate x MAXMIN passes it)
#                CONTAINER_GB (120; 150 for gemma, as in launch_distances.sh)
#                CLOUD (SECURE; COMMUNITY for l8: the guard caps an A40 at $0.50/h, a community A40 costs ~$0.35 and a secure one
#                      ~$0.59, and retry.sh only retries "no stock", not a refused price)
#                LIMIT  only the first N branches (the histories stay complete)
#                OUT  results dir, repo-relative and without `..` (persona_affect/a8_rejection/results/$M); the launcher pulls
#                     /root/NEEL/$OUT back from the pod, and its output is also tee'd to $OUT.launch.log
#                NAME  pod name and ledger note (pa-a8-$M)    DRY_RUN  print the launcher command and stop
# HF_TOKEN comes from .env (a line HF_TOKEN=...), else from the environment. g3 and l8 are gated models, so an empty token
# refuses the launch here instead of failing at the download, 15 minutes into a pod.
# Ships persona_affect, welfare_training/stage0.py (+ its __init__.py), vigilante/pod and infra, and pulls $OUT back.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
: "${M:?set M (g3, gemma, q27 or l8)}"
GATED=""
case "$M" in
  g3)    DEF_GPUS="NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe" DEF_GB=120 DEF_MIN=100 DEF_CLOUD=SECURE GATED=1 ;;
  gemma) DEF_GPUS="NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe" DEF_GB=150 DEF_MIN=150 DEF_CLOUD=SECURE ;;
  q27)   DEF_GPUS="NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe" DEF_GB=120 DEF_MIN=100 DEF_CLOUD=SECURE ;;
  l8)    DEF_GPUS="NVIDIA A40|NVIDIA L40S" DEF_GB=120 DEF_MIN=100 DEF_CLOUD=COMMUNITY GATED=1 ;;
  *) echo "M must be g3, gemma, q27 or l8, got '$M'" >&2; exit 2 ;;
esac
MAXMIN=${MAXMIN:-$DEF_MIN}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'" >&2; exit 2 ;; esac
MAXMIN=$((10#$MAXMIN))
[ "$MAXMIN" -ge 1 ] || { echo "MAXMIN must be at least 1 minute" >&2; exit 2; }
if [ -n "${LIMIT:-}" ]; then
  case "$LIMIT" in *[!0-9]*) echo "LIMIT must be a whole number of branches, got '$LIMIT'" >&2; exit 2 ;; esac
  LIMIT=$((10#$LIMIT))
  [ "$LIMIT" -ge 1 ] || { echo "LIMIT must be at least 1 branch" >&2; exit 2; }
fi
OUT=${OUT:-persona_affect/a8_rejection/results/$M}
case "$OUT" in
  /*|..|../*|*/..|*/../*) echo "OUT must be a path inside the repo, without '..' (it is pulled back from /root/NEEL/\$OUT on the pod), got '$OUT'" >&2; exit 2 ;;
esac
OUT=${OUT%/}
# the token, without uv: .env wins over the environment, as in the sibling launchers; comments and empty values do not count
if [ -f .env ]; then
  ENV_TOKEN=$(sed -n -E 's/^[[:space:]]*(export[[:space:]]+)?HF_TOKEN[[:space:]]*=[[:space:]]*["'\'']?([^"'\''[:space:]]+).*$/\2/p' .env | head -n 1 || true)
  if [ -n "$ENV_TOKEN" ]; then HF_TOKEN=$ENV_TOKEN; fi
fi
HF_TOKEN=${HF_TOKEN:-}
if [ -n "$GATED" ] && [ -z "$HF_TOKEN" ]; then
  echo "HF_TOKEN is empty (no HF_TOKEN= line in .env, none in the environment) and M=$M is a gated model: the pod could not download it" >&2
  exit 2
fi
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
IFS='|' read -ra GPUARR <<< "${GPUS:-$DEF_GPUS}"
NAME=${NAME:-pa-a8-$M}
PODENV=(M="$M" OUT="$OUT")
if [ -n "${LIMIT:-}" ]; then PODENV+=(LIMIT="$LIMIT"); fi
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "${GPUARR[@]}" --container-gb "${CONTAINER_GB:-$DEF_GB}" --cloud "${CLOUD:-$DEF_CLOUD}"
     --ship persona_affect --ship welfare_training/__init__.py --ship welfare_training/stage0.py --ship vigilante/pod --ship infra
     --results-dir "$OUT" -- env "${PODENV[@]}" bash persona_affect/a8_rejection/pod.sh)
if [ -n "${DRY_RUN:-}" ]; then echo "${CMD[*]}"; exit 0; fi

export HF_TOKEN
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
# Ctrl-C (or a kill) must not skip the booking: trap the signals here so the script goes on to the ledger, and keep tee
# alive through INT, because the launcher's cleanup (terminate the pod, print the final cost line) comes after the signal
trap 'echo "[launch] interrupted: booking what the launcher log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT; exec tee "$LOG")
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
  uv run python -c 'import sys; from persona_affect.common.util import ledger_add; ledger_add(sys.argv[1], "gpu", float(sys.argv[2]), note=sys.argv[3])' A8 "$USD" "$NOTE" \
    && echo "[ledger] A8 gpu \$$USD ($NOTE)" \
    || echo "[ledger] FAILED to book \$$USD for A8 ($NOTE): add it with persona_affect.common.util.ledger_add"
fi
exit "$rc"
