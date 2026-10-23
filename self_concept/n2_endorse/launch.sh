#!/usr/bin/env bash
# SC-N2 launcher: one guarded A100 80GB pod (pod.sh), P11f's launcher pattern, booked to self_concept/ledger.jsonl.
#   bash self_concept/n2_endorse/launch.sh                                   # MAXMIN (190)
#   no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash self_concept/n2_endorse/launch.sh
# Name sc-n2, results self_concept/n2_endorse/results/main. GPUs: A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY first
# (the launcher falls back to SECURE); MAXCOST = ceil(MAXMIN x $1.80 / 60) to the cent. V1's VN_s1 adapter goes
# straight from V1's folder as --sync-extra files (no copy or link in this folder, so no other session's code sync can
# pick it up) and the pod reads it there (AD). Every data/adapters* folder at any depth stays out of the code sync.
# Ships V1's ship.txt + self_concept/__init__.py + self_concept/n2_endorse (with data/rows). Refuses unless the tag sc-n2-prereg exists, the rows match
# data/rows/sha256.json, and check_cap("SC-N2", 8, planned = the upper bound) passes. SC_N2_LAUNCH_DRY=1 prints it.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/n2_endorse; V=persona_affect/v1_value_hook
DRY=${SC_N2_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[sc-n2-launch] refusing to launch a pod under pytest (SC_N2_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=sc-n2; RES=$H/results/main; MAXMIN=${MAXMIN:-190}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship "$H")
AD=$V/data/adapters
EXTRA=(--exclude 'data/adapters*')   # bsdtar matches path tails: */data/adapters* at any depth, made now or later
for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$AD/VN_s1/$f"); done
for d in persona_affect/*/jobs; do [ -d "$d" ] && EXTRA+=(--exclude "$d"); done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" AD="$AD" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[sc-n2-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-n2-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-n2-prereg: commit and tag DESIGN.md, the rows and the code first"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
d = Path('$H/data/rows'); want = json.loads((d / 'sha256.json').read_text())
bad = [a for a, s in want.items() if hashlib.sha256((d / f'{a}.jsonl.gz').read_bytes()).hexdigest() != s]
sys.exit(f'[sc-n2-launch] rows changed since the prereg: {bad}' if bad else 0)"
for f in adapter_model.safetensors adapter_config.json; do
  [ -s "$AD/VN_s1/$f" ] || { echo "[sc-n2-launch] missing $AD/VN_s1/$f (V1's adapter)"; exit 2; }
done
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-N2", 8.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-N2's ledger plus this pod's upper bound \$$BOUND would pass the \$8 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-N2 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
