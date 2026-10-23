#!/usr/bin/env bash
# PA-P11e launcher: one guarded A100 80GB pod (pod.sh), P11's launcher pattern.
#   bash persona_affect/p11e_mirror/launch.sh                                 # MAXMIN (100)
#   no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona_affect/p11e_mirror/launch.sh
# Name pa-p11e, results persona_affect/p11e_mirror/results/main. GPUs: A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY
# first (the launcher falls back to SECURE), 150 GB container; MAXCOST = ceil(MAXMIN x $1.80 / 60) to the cent.
# V1's three seed-1 adapters are hard-linked into data/adapters (no extra disk) and go as --sync-extra files; every
# mini's jobs/ and data/adapters folder stays out of the code sync (launch_gen.sh's rule). Refuses unless the tag
# pa-p11e-prereg exists and check_cap("P11e", 4, planned = the upper bound) passes. P11E_LAUNCH_DRY=1 prints it.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/p11e_mirror; V=persona_affect/v1_value_hook
DRY=${P11E_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[p11e-launch] refusing to launch a pod under pytest (P11E_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=pa-p11e; RES=$H/results/main; MAXMIN=${MAXMIN:-100}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
src() {   # src ADAPTER -> where V1 keeps it
  case "$1" in VN_s1) echo "$V/data/adapters/VN_s1" ;; N?_s1) echo "$V/results/full_s1/ad_later/$1" ;; esac
}
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
EXTRA=()
for a in VN_s1 NW_s1 NZ_s1; do
  for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$H/data/adapters/$a/$f"); done
done
EXTRA+=(--exclude "$H/data/adapters")   # linked below, after this list is built
for d in persona_affect/*/jobs persona_affect/*/data/adapters; do
  [ -d "$d" ] && [ "$d" != "$H/data/adapters" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[p11e-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/pa-p11e-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag pa-p11e-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
for a in VN_s1 NW_s1 NZ_s1; do
  mkdir -p "$H/data/adapters/$a"
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$(src "$a")/$f" ] || { echo "[p11e-launch] missing $(src "$a")/$f (V1's adapter)"; exit 2; }
    ln -f "$(src "$a")/$f" "$H/data/adapters/$a/$f"
  done
done
BOUND=$(bash persona_affect/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from persona_affect.common.util import check_cap; check_cap("P11e", 4.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: P11e's ledger plus this pod's upper bound \$$BOUND would pass the \$4 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash persona_affect/common/book_pod.sh "$LOG" P11e "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
