#!/usr/bin/env bash
# PA-V1b launcher: one guarded A100 80GB pod (pod.sh), V1's launcher pattern (v1_value_hook/launch.sh).
#   bash persona_affect/v1b_dose/launch.sh                                   # MAXMIN (210)
#   no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona_affect/v1b_dose/launch.sh
# Name pa-v1b, results persona_affect/v1b_dose/results/main. GPUs: A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY first
# (the launcher falls back to SECURE), 150 GB container, RUNPOD_ANY_CUDA=1; MAXCOST = ceil(MAXMIN x $1.80 / 60) to the
# cent. Ships V1's ship.txt; V1's VN_s1 adapter goes as --sync-extra files (results/ never ships).
# Refuses unless the tag pa-v1b-prereg exists, every data/rows/*.jsonl.gz sha256 is in DESIGN.md, and
# check_cap("V1b", 8, planned = the pod's upper bound) passes. V1B_LAUNCH_DRY=1 prints the command and stops. The pod
# is booked to persona_affect/ledger.jsonl by common/book_pod.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/v1b_dose; V=persona_affect/v1_value_hook
DRY=${V1B_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[v1b-launch] refusing to launch a pod under pytest (V1B_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=pa-v1b; RES=$H/results/main; MAXMIN=${MAXMIN:-210}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
EXTRA=()
for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$V/data/adapters/VN_s1/$f"); done
EXTRA+=(--exclude "$V/data/adapters")
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[v1b-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/pa-v1b-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag pa-v1b-prereg: commit and tag DESIGN.md and data/rows first"; exit 2; }
for p in "$H"/data/rows/*.jsonl.gz; do
  sha=$(shasum -a 256 "$p" | cut -d' ' -f1)
  grep -q "$sha" "$H/DESIGN.md" || { echo "[v1b-launch] $p sha256 $sha is not in $H/DESIGN.md"; exit 2; }
done
for f in adapter_model.safetensors adapter_config.json; do
  [ -s "$V/data/adapters/VN_s1/$f" ] || { echo "[v1b-launch] missing $V/data/adapters/VN_s1/$f (V1's adapter)"; exit 2; }
done
BOUND=$(bash persona_affect/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from persona_affect.common.util import check_cap; check_cap("V1b", 8.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: V1b's ledger plus this pod's upper bound \$$BOUND would pass the \$8 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash persona_affect/common/book_pod.sh "$LOG" V1b "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
