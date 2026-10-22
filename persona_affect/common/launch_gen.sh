#!/usr/bin/env bash
# One guarded RunPod pod for a persona_affect mini: vigilante's vLLM episode runner on a jobs.jsonl, with the mini's cases and
# conds registered first (pod_gen.sh -> pod_gen.py). Mirrors persona/inference/launch_distances.sh. When the launcher returns,
# the pod's cost goes to persona_affect/ledger.jsonl (kind gpu): the last `elapsed cost ~$X` line it printed.
#   MINI=P12 NAME=pa-p12-27b JOBS=persona_affect/p12_named_traits/jobs/cal_qwen.jsonl OUT=persona_affect/p12_named_traits/out/cal_qwen \
#     REGISTER=persona_affect.p12_named_traits.cases MAXMIN=40 bash persona_affect/common/launch_gen.sh
#   # Gemma 4: add MODEL=google/gemma-4-31B-it FMT=gemma4 MML=12288 CONTAINER_GB=150 (and "fmt": "gemma4" in every job)
#   # no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona_affect/common/launch_gen.sh
# env, required: MINI  ledger id (P12, A4, ...)
#                JOBS  jobs.jsonl, repo-relative, under persona_affect/ (only that is shipped), and not under a directory
#                      called results, logs, prep or reference: the launcher's sync drops those
#                OUT   results dir, repo-relative; pulled back when the job ends. The launcher's own output is tee'd to $OUT.launch.log
#      optional: MODEL (Qwen/Qwen3.6-27B)  NAME pod name and ledger note (pa-$MINI)  REGISTER space-separated modules that call
#                vig.register  FMT (qwen)  MML (40960)  GPUS pipe-separated (NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe)
#                MAXMIN (90)  CONTAINER_GB (120)  CLOUD (COMMUNITY)
# MAXCOST = MAXMIN x $1.80/h. The guard refuses a pod whose rate x MAXMIN passes it, so size MAXMIN to the mini's cap.
# Each job's "fmt" must equal FMT: the episode parser reads the job's field, the template conversion reads the flag.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
: "${MINI:?set MINI (the id in persona_affect/ledger.jsonl, e.g. P12)}" "${JOBS:?set JOBS (jobs.jsonl under persona_affect/)}" "${OUT:?set OUT (results dir)}"
MODEL=${MODEL:-Qwen/Qwen3.6-27B} NAME=${NAME:-pa-$MINI} REGISTER=${REGISTER:-} FMT=${FMT:-qwen} MML=${MML:-40960} MAXMIN=${MAXMIN:-90}
JOBS=${JOBS#./} OUT=${OUT%/}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXMIN=$((10#$MAXMIN))
case "$JOBS" in persona_affect/*) ;; *) echo "JOBS must be a repo-relative path under persona_affect/ (only that is shipped): $JOBS"; exit 2;; esac
case "/$JOBS/" in */results/*|*/logs/*|*/prep/*|*/reference/*)
  echo "JOBS sits under a results/, logs/, prep/ or reference/ directory, which the launcher's sync drops: $JOBS"; exit 2;; esac
[ -f "$JOBS" ] || { echo "JOBS not found: $JOBS"; exit 2; }
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
if [ -n "${GPUS:-}" ]; then IFS='|' read -ra GPUARR <<< "$GPUS"; else GPUARR=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); fi
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
# Ctrl-C (or a kill) must not skip the booking: trap the signals here so the script goes on to the ledger, and keep tee
# alive through INT, because the launcher's cleanup (terminate the pod, print the final cost line) comes after the signal
trap 'echo "[launch] interrupted: booking what the launcher log shows"' INT TERM HUP
uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "${GPUARR[@]}" --container-gb "${CONTAINER_GB:-120}" --cloud "${CLOUD:-COMMUNITY}" \
  --ship persona_affect --ship vigilante/src --ship vigilante/cases --ship vigilante/pod --ship vigilante/__init__.py \
  --ship infra --results-dir "$OUT" \
  -- env MODEL="$MODEL" JOBS="$JOBS" OUT="$OUT" REGISTER="$REGISTER" FMT="$FMT" MML="$MML" bash persona_affect/common/pod_gen.sh \
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
