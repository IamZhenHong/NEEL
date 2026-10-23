#!/usr/bin/env bash
# PA-P11c launcher: one guarded A100 80GB pod (pod.sh), P11's launcher pattern.
#   bash persona_affect/p11c_scale/launch.sh                                  # MAXMIN (280)
#   no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona_affect/p11c_scale/launch.sh
# Name pa-p11c, results persona_affect/p11c_scale/results/main. GPUs: A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY
# first (the launcher falls back to SECURE), 250 GB container (both snapshots + one merged 27B); MAXCOST =
# ceil(MAXMIN x $1.80 / 60) to the cent. No adapters are shipped: every stage trains on the pod from frozen rows (P11's
# p11_saydo/data/rows, V1's v1_value_hook/data/rows, both shipped with persona_affect). Every persona_affect
# data/adapters dir and the large jobs files of other minis are excluded. Refuses unless the tag pa-p11c-prereg exists,
# the rows match DESIGN.md and check_cap("P11c", 10, planned = the upper bound) passes. P11C_LAUNCH_DRY=1 prints it.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=persona_affect/p11c_scale; V=persona_affect/v1_value_hook
DRY=${P11C_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[p11c-launch] refusing to launch a pod under pytest (P11C_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=pa-p11c; RES=$H/results/main; MAXMIN=${MAXMIN:-280}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
EXTRA=(--exclude "persona_affect/*/data/adapters" --exclude "$V/data/adapters" --exclude persona_affect/p11_saydo/data/adapters
       --exclude persona_affect/v1c_selfreport/data/adapters --exclude persona_affect/p11b_voice/data/adapters)
for j in c2_capaware p13_deference p9_bakeoff a1_frustration; do EXTRA+=(--exclude "persona_affect/$j/jobs"); done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 250
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[p11c-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/pa-p11c-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag pa-p11c-prereg: commit and tag DESIGN.md first"; exit 2; }
for p in persona_affect/p11_saydo/data/rows/{SAYU,SAYR,PLAC}.jsonl.gz "$V"/data/rows/{W,Z}.jsonl.gz; do
  sha=$(shasum -a 256 "$p" | cut -d' ' -f1)
  grep -q "$sha" "$H/DESIGN.md" || { echo "[p11c-launch] $p sha256 $sha is not in $H/DESIGN.md"; exit 2; }
done
BOUND=$(bash persona_affect/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from persona_affect.common.util import check_cap; check_cap("P11c", 10.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: P11c's ledger plus this pod's upper bound \$$BOUND would pass the \$10 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash persona_affect/common/book_pod.sh "$LOG" P11c "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
