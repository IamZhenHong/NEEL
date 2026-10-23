#!/usr/bin/env bash
# SC-B5 launcher: one guarded A100 80GB pod (pod.sh), booked to self_concept/ledger.jsonl.
#   bash self_concept/b5_dose/launch.sh                                      # MAXMIN (99)
#   no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash self_concept/b5_dose/launch.sh
# Name sc-b5, results self_concept/b5_dose/results/main. GPUs: A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY;
# MAXCOST = ceil(MAXMIN x $1.39 / 60) to the cent, so the guard refuses any pod dearer than $1.39/hr and the upper
# bound with the 30-min self-destruct stays within the $3 cap. V1's VN_s1 and NZ_s1 go straight from V1's folders as
# --sync-extra files (nothing linked here); every data/adapters* folder at any depth stays out of the code sync.
# Ships V1's ship.txt + self_concept/__init__.py + self_concept/b5_dose + SC-N2's train_ck.py (the checkpoint driver).
# Refuses unless the tag sc-b5-prereg exists and check_cap("SC-B5", 3, planned = the upper bound) passes.
# SC_B5_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/b5_dose; V=persona_affect/v1_value_hook
VNAD=$V/data/adapters; NZAD=$V/results/full_s1/ad_later
DRY=${SC_B5_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[sc-b5-launch] refusing to launch a pod under pytest (SC_B5_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=sc-b5; RES=$H/results/main; MAXMIN=${MAXMIN:-99}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 139 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship "$H" --ship self_concept/n2_endorse/__init__.py
       --ship self_concept/n2_endorse/train_ck.py)
EXTRA=(--exclude 'data/adapters*')   # bsdtar matches path tails: */data/adapters* at any depth, made now or later
for p in "$VNAD/VN_s1" "$NZAD/NZ_s1"; do
  for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$p/$f"); done
done
for d in persona_affect/*/jobs; do [ -d "$d" ] && EXTRA+=(--exclude "$d"); done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" VNAD="$VNAD" NZAD="$NZAD" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[sc-b5-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-b5-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-b5-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
for p in "$VNAD/VN_s1" "$NZAD/NZ_s1"; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$p/$f" ] || { echo "[sc-b5-launch] missing $p/$f (V1's adapter)"; exit 2; }
  done
done
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-B5", 3.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-B5's ledger plus this pod's upper bound \$$BOUND would pass the \$3 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-B5 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
