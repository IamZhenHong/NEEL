#!/usr/bin/env bash
# SC-A5 launcher: one guarded A100 80GB pod (pod.sh), V1's launcher pattern.
#   bash self_concept/a5_coupled/launch.sh                                   # MAXMIN (110)
#   no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash self_concept/a5_coupled/launch.sh
# Name sc-a5, results self_concept/a5_coupled/results/main. GPUs: A100-SXM4-80GB or A100 80GB PCIe, SECURE
# (a slow COMMUNITY host stalled earlier pods), 150 GB container; MAXCOST = ceil(MAXMIN x $1.80 / 60) to the cent.
# V1's VN_s1 adapter (from v1_value_hook/data/adapters) is hard-linked into data/adapters (no extra disk) and goes as
# --sync-extra files; every mini's jobs/ and data/adapters* folders stay off the code sync (launch_gen.sh's rule).
# Refuses unless the tag sc-a5-prereg
# exists, data/rows matches DESIGN.md and check_cap("SC-A5", 5, planned = the upper bound) passes. A5_LAUNCH_DRY=1 prints it.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/a5_coupled; V=persona_affect/v1_value_hook
DRY=${A5_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[a5-launch] refusing to launch a pod under pytest (A5_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=sc-a5; RES=$H/results/main; MAXMIN=${MAXMIN:-110}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
src() {   # src ADAPTER -> where V1 keeps it
  case "$1" in VN_s1) echo "$V/data/adapters/VN_s1" ;; VN_s2) echo "$V/results/full_s2/ad/VN_s2" ;;
               N?_s1) echo "$V/results/full_s1/ad_later/$1" ;; N?_s2) echo "$V/results/full_s2/ad_later/$1" ;; esac
}
SHIP=(--ship self_concept)
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
EXTRA=()
for a in VN_s1; do
  for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$H/data/adapters/$a/$f"); done
done
for d in persona_affect/*/jobs persona_affect/*/data/adapters* self_concept/*/data/adapters*; do [ -d "$d" ] && EXTRA+=(--exclude "$d"); done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud SECURE --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[a5-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-a5-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-a5-prereg: commit and tag DESIGN.md, rows and items.py first"; exit 2; }
for p in "$H"/data/rows/*.jsonl.gz; do
  sha=$(shasum -a 256 "$p" | cut -d' ' -f1)
  grep -q "$sha" "$H/DESIGN.md" || { echo "[a5-launch] $p sha256 $sha is not in $H/DESIGN.md"; exit 2; }
done
for a in VN_s1; do
  mkdir -p "$H/data/adapters/$a"
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$(src "$a")/$f" ] || { echo "[a5-launch] missing $(src "$a")/$f (V1's adapter)"; exit 2; }
    ln -f "$(src "$a")/$f" "$H/data/adapters/$a/$f"
  done
done
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; import self_concept.common.ledger; from self_concept.common.ledger import check_cap; check_cap("SC-A5", 5.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-A5's ledger plus this pod's upper bound \$$BOUND would pass the \$3 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-A5 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
