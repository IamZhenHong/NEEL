#!/usr/bin/env bash
# PA-P11e fill launcher: one guarded A100 80GB pod (pod_fill.sh) for a cell the main pod's deadline stop skipped.
#   bash persona_affect/p11e_mirror/launch_fill.sh                            # MAXMIN (45), B=NZ ARM=SHUF
# Name pa-p11e-fill, results persona_affect/p11e_mirror/results/fill. Ships V1's VN_s1 and ${B}_s1 (data/adapters, linked
# by launch.sh) and the cell's own adapter (data/adapters_nz/N${ARM}_s1) as --sync-extra files. Same guards as launch.sh:
# the tag pa-p11e-prereg, check_cap("P11e", 4, planned = the upper bound). P11E_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/p11e_mirror; V=persona_affect/v1_value_hook
DRY=${P11E_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[p11e-fill-launch] refusing to launch a pod under pytest (P11E_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=pa-p11e-fill; RES=$H/results/fill; MAXMIN=${MAXMIN:-45}; B=${B:-NZ}; ARM=${ARM:-SHUF}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
EXTRA=()
for p in "$H/data/adapters/VN_s1" "$H/data/adapters/${B}_s1" "$H/data/adapters_nz/N${ARM}_s1"; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$p/$f" ] || { echo "[p11e-fill-launch] missing $p/$f"; exit 2; }
    EXTRA+=(--sync-extra "$p/$f")
  done
done
EXTRA+=(--exclude "$H/data/adapters" --exclude "$H/data/adapters_nz")
for d in persona_affect/*/jobs persona_affect/*/data/adapters; do
  [ -d "$d" ] && [ "$d" != "$H/data/adapters" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" B="$B" ARM="$ARM"
     bash "$H/pod_fill.sh")
if [ "$DRY" = 1 ]; then
  echo "[p11e-fill-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/pa-p11e-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag pa-p11e-prereg"; exit 2; }
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
