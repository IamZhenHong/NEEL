#!/usr/bin/env bash
# PA-V1 launcher (plans/V1.md Task 7): one guarded A100 80GB pod per run.
#   bash persona_affect/v1_value_hook/launch.sh pilot                       # REGIME (R1) D (d1) LADDER (1)
#   REGIME=R2 ROWS=data/rows D=<rung> LADDER=0 BASE_DIR=results/pilot bash persona_affect/v1_value_hook/launch.sh pilot_r2
#   SEED=1|2 REGIME=R1 DOSE=<rung> bash persona_affect/v1_value_hook/launch.sh full
#   no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona_affect/v1_value_hook/launch.sh pilot
# Names and results dirs: pa-v1-pilot / results/pilot, pa-v1-pilot-r2 / results/pilot_r2, pa-v1-full-s{SEED} /
# results/full_s{SEED}. GPUs: A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY first (the launcher falls back to SECURE),
# 150 GB container, RUNPOD_ANY_CUDA=1; MAXCOST = ceil(MAXMIN x $1.80 / 60) to the cent; MAXMIN defaults: pilot 220,
# pilot_r2 250, full s1 205, full s2 289. Ships each path in ship.txt (the launcher drops every results/ dir).
# The pod gets its settings through `env K=V ... bash <pod script>` (the local env does not reach it over ssh).
# Pod code never reads results/ (not synced), so this script stages what a pod needs from there under data/ first:
#   pilot_r2   BASE_DIR (default results/pilot) / eval_base.jsonl -> data/pilot_base/ (shipped); the pod reads that
#   full s1    the pilot's value adapters ADAPTERS_FROM (results/pilot/ad, or results/pilot_r2/ad when REGIME=R2)
#              /V{N,J,S}_s1 -> data/adapters/ (kept out of the tarball; each adapter_model.safetensors and
#              adapter_config.json is a separate --sync-extra file)
# Refuses unless check_cap("V1", 18, planned = the pod's upper bound) passes and, for full, every data/rows/*.jsonl.gz
# sha256 appears in DESIGN.md. Refuses under pytest unless V1_LAUNCH_DRY=1, which prints the launcher command (and
# stages nothing, checks nothing). The pod is booked to persona_affect/ledger.jsonl by common/book_pod.sh, as
# common/launch_gen.sh does: the cost on the launcher's terminated line, the upper bound if a pod line has none,
# nothing if no pod was created.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/v1_value_hook
MODE=${1:?usage: launch.sh pilot|pilot_r2|full}
DRY=${V1_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[v1-launch] refusing to launch a pod under pytest (V1_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
rel() { case "$1" in "$H"/*) echo "$1" ;; *) echo "$H/$1" ;; esac; }   # data/rows -> persona_affect/v1_value_hook/data/rows
EXTRA=()
case "$MODE" in
  pilot)
    NAME=pa-v1-pilot; RES=$H/results/pilot; MAXMIN=${MAXMIN:-220}; SCRIPT=$H/pod_pilot.sh
    ENVS=(REGIME="${REGIME:-R1}" D="${D:-d1}" LADDER="${LADDER:-1}" OUT="$RES")
    [ -z "${ROWS:-}" ] || ENVS+=(ROWS="$(rel "$ROWS")") ;;
  pilot_r2)
    NAME=pa-v1-pilot-r2; RES=$H/results/pilot_r2; MAXMIN=${MAXMIN:-250}; SCRIPT=$H/pod_pilot.sh
    : "${D:?set D (the rung chosen by the R1 pilot)}"
    BASE_FROM=$(rel "${BASE_DIR:-results/pilot}")
    ENVS=(REGIME="${REGIME:-R2}" ROWS="$(rel "${ROWS:-data/rows}")" D="$D" LADDER="${LADDER:-0}"
          BASE_DIR="$H/data/pilot_base" OUT="$RES") ;;
  full)
    : "${SEED:?set SEED (1 or 2)}" "${DOSE:?set DOSE (the frozen Warden rung)}"
    NAME=pa-v1-full-s$SEED; RES=$H/results/full_s$SEED; SCRIPT=$H/pod_full.sh
    if [ "$SEED" = 1 ]; then MAXMIN=${MAXMIN:-205}; else MAXMIN=${MAXMIN:-289}; fi
    ENVS=(SEED="$SEED" REGIME="${REGIME:-R1}" DOSE="$DOSE" OUT="$RES")
    if [ "$SEED" = 1 ]; then
      for v in N J S; do for f in adapter_model.safetensors adapter_config.json; do
        EXTRA+=(--sync-extra "$H/data/adapters/V${v}_s1/$f")
      done; done
      EXTRA+=(--exclude "$H/data/adapters")
    fi ;;
  *) echo "usage: launch.sh pilot|pilot_r2|full"; exit 2 ;;
esac
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$H/ship.txt"
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" ${EXTRA[@]+"${EXTRA[@]}"} --results-dir "$RES" -- env "${ENVS[@]}" bash "$SCRIPT")
if [ "$DRY" = 1 ]; then
  echo "[v1-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
# ---- staging (local reads of results/ only) and refusals
if [ "$MODE" = pilot_r2 ]; then
  [ -s "$BASE_FROM/eval_base.jsonl" ] || { echo "[v1-launch] no $BASE_FROM/eval_base.jsonl (the R1 pilot's base rows)"; exit 2; }
  mkdir -p "$H/data/pilot_base" && cp "$BASE_FROM/eval_base.jsonl" "$H/data/pilot_base/eval_base.jsonl"
fi
if [ "$MODE" = full ]; then
  for f in J S N W E Z; do
    p=$H/data/rows/$f.jsonl.gz
    [ -s "$p" ] || { echo "[v1-launch] $p missing (Task 9 freezes the rows)"; exit 2; }
    sha=$(shasum -a 256 "$p" | cut -d' ' -f1)
    grep -q "$sha" "$H/DESIGN.md" 2>/dev/null || { echo "[v1-launch] $p sha256 $sha is not in $H/DESIGN.md"; exit 2; }
  done
  if [ "$SEED" = 1 ]; then
    FROM=$(rel "${ADAPTERS_FROM:-results/$([ "${REGIME:-R1}" = R2 ] && echo pilot_r2 || echo pilot)/ad}")
    for v in N J S; do
      mkdir -p "$H/data/adapters/V${v}_s1"
      for f in adapter_model.safetensors adapter_config.json; do
        [ -s "$FROM/V${v}_s1/$f" ] || { echo "[v1-launch] missing $FROM/V${v}_s1/$f (the pilot's adapter)"; exit 2; }
        cp "$FROM/V${v}_s1/$f" "$H/data/adapters/V${v}_s1/$f"
      done
    done
  fi
fi
BOUND=$(bash persona_affect/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from persona_affect.common.util import check_cap; check_cap("V1", 18.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: V1's ledger plus this pod's upper bound \$$BOUND would pass the \$18 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash persona_affect/common/book_pod.sh "$LOG" V1 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
