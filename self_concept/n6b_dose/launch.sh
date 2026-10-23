#!/usr/bin/env bash
# SC-N6b launcher: one guarded A100 80GB pod (pod.sh), SC-N4's launcher with one stage.
#   bash self_concept/n6b_dose/launch.sh        # MAXMIN 120 (bound $4.50 under the $5 cap)
# Name sc-n6b, results self_concept/n6b_dose/results/main. GPUs: A100-SXM4-80GB or A100 80GB PCIe, SECURE, 150 GB
# container; MAXCOST = ceil(MAXMIN x $1.80 / 60) to the cent. V1's VN_s1 adapter is hard-linked into data/adapters and
# goes as --sync-extra files; every jobs/ and data/adapters* folder stays off the code sync. Refuses without the tag
# sc-n6b-prereg, if V1's W / Z rows' sha256 is not in DESIGN.md, or when the SC-N6b ledger plus this pod's upper
# bound would pass the $5 cap. Books the pod to self_concept/ledger.jsonl. N6B_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/n6b_dose; V=persona_affect/v1_value_hook
DRY=${N6B_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[n6b-launch] refusing to launch a pod under pytest (N6B_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=sc-n6b; RES=$H/results/main; MAXMIN=${MAXMIN:-120}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship "$H")
EXTRA=()
for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$H/data/adapters/VN_s1/$f"); done
for d in persona_affect/*/jobs persona_affect/*/data/adapters* "$H/data/adapters" "$H/results"; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
EXTRA+=(--exclude 'data/adapters*')   # bsdtar matches path tails: also drops adapter folders made after this glob ran
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud SECURE --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[n6b-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-n6b-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-n6b-prereg: commit and tag DESIGN.md and items.py first"; exit 2; }
for p in persona_affect/v1_value_hook/data/rows/{W,Z}.jsonl.gz; do
  sha=$(shasum -a 256 "$p" | cut -d' ' -f1)
  grep -q "$sha" "$H/DESIGN.md" || { echo "[n6b-launch] $p sha256 $sha is not in $H/DESIGN.md"; exit 2; }
done
mkdir -p "$H/data/adapters/VN_s1"
for f in adapter_model.safetensors adapter_config.json; do
  [ -s "$V/data/adapters/VN_s1/$f" ] || { echo "[n6b-launch] missing $V/data/adapters/VN_s1/$f (V1's adapter)"; exit 2; }
  ln -f "$V/data/adapters/VN_s1/$f" "$H/data/adapters/VN_s1/$f"
done
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; import self_concept.common.ledger; from self_concept.common.ledger import check_cap; check_cap("SC-N6b", 5.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-N6b's ledger plus this pod's upper bound \$$BOUND would pass the \$5 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-N6b "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
