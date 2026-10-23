#!/usr/bin/env bash
# SC-G1 launcher: one guarded A100 80GB pod (pod.sh), P15's launcher pattern, booked to self_concept/ledger.jsonl.
#   bash self_concept/g1_self/launch.sh                 # MAXMIN (80)
#   SC_G1_LAUNCH_DRY=1 bash self_concept/g1_self/launch.sh
# Uploads code only: V1's seed-1 stages are trained on the pod (rebuild.sh V1_ON_POD=1, AD=/root/v1). Refuses unless the tag
# sc-g1-prereg exists and check_cap("SC-G1", 4.5, planned = the upper bound) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/g1_self; V=persona_affect/v1_value_hook
DRY=${SC_G1_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[g1-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
NAME=sc-g1${G1_MODELS:+-nz}; RES=$H/results/${G1_RES:-main}; MAXMIN=${MAXMIN:-80}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
src() { case "$1" in VN_s1) echo "$V/data/adapters/VN_s1" ;; N?_s1) echo "$V/results/full_s1/ad_later/$1" ;; esac; }
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$H")
EXTRA=(--exclude "$H/results" --exclude 'data/adapters*')   # nothing uploaded but code: V1's stages train on the pod   # any depth, incl. folders made mid-upload (peer's 405f8a1 lesson); --sync-extra still ships V1's
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 200
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" V1_ON_POD=1 AD=/root/v1 G1_MODELS="${G1_MODELS:-}" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[g1-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-g1-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-g1-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-G1", 4.5, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-G1's ledger plus this pod's upper bound \$$BOUND would pass the \$4.5 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-G1 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
