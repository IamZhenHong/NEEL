#!/usr/bin/env bash
# SC-D2 + SC-C3 launcher: one guarded A100 80GB pod (pod.sh), P17's launcher pattern.
#   bash self_concept/d2_resets/launch.sh                                          # MAXMIN (100)
# Name sc-d2c3, results self_concept/d2_resets/results/main (C3's in main/c3, moved to self_concept/c3_selfperception/
# results/main after the pull). V1's seed-1 adapters (VN_s1 from v1_value_hook/data/adapters, NW_s1 / NZ_s1 from
# results/full_s1/ad_later, as P15 / P17) are hard-linked into data/adapters and go as --sync-extra files. Refuses unless
# the tags sc-d2-prereg and sc-c3-prereg exist, P11's SAYU / PLAC rows and C3's pool.json match the DESIGN files, and the
# caps hold: SC-D2's ledger + the pod's upper bound <= $4, and SC-D2 + SC-C3 + the bound <= $5. Books with book.py
# (split by eval seconds). D2_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/d2_resets; C=self_concept/c3_selfperception; V=persona_affect/v1_value_hook
DRY=${D2_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[d2-launch] refusing to launch a pod under pytest (D2_LAUNCH_DRY=1 prints the command instead)"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=sc-d2c3; RES=$H/results/main; MAXMIN=${MAXMIN:-100}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
src() { case "$1" in VN_s1) echo "$V/data/adapters/VN_s1" ;; N?_s1) echo "$V/results/full_s1/ad_later/$1" ;; esac; }
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$H" --ship "$C")
EXTRA=()
for a in VN_s1 NW_s1 NZ_s1; do
  for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$H/data/adapters/$a/$f"); done
done
EXTRA+=(--exclude "$H/data/adapters" --exclude "$H/results" --exclude "$C/results")
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results; do [ -d "$d" ] && EXTRA+=(--exclude "$d"); done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[d2-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
for t in sc-d2-prereg sc-c3-prereg; do
  git rev-parse -q --verify "refs/tags/$t" > /dev/null 2>&1 || { echo "[prereg] no tag $t: commit and tag first"; exit 2; }
done
for p in persona_affect/p11_saydo/data/rows/SAYU.jsonl.gz persona_affect/p11_saydo/data/rows/PLAC.jsonl.gz; do
  sha=$(shasum -a 256 "$p" | cut -d' ' -f1)
  grep -q "$sha" "$H/DESIGN.md" || { echo "[d2-launch] $p sha256 $sha is not in $H/DESIGN.md"; exit 2; }
done
sha=$(shasum -a 256 "$C/data/pool.json" | cut -d' ' -f1)
grep -q "$sha" "$C/DESIGN.md" || { echo "[d2-launch] $C/data/pool.json sha256 $sha is not in $C/DESIGN.md"; exit 2; }
for a in VN_s1 NW_s1 NZ_s1; do
  mkdir -p "$H/data/adapters/$a"
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$(src "$a")/$f" ] || { echo "[d2-launch] missing $(src "$a")/$f (V1's adapter)"; exit 2; }
    ln -f "$(src "$a")/$f" "$H/data/adapters/$a/$f"
  done
done
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap, ledger_total
b = float(sys.argv[1]); check_cap("SC-D2", 4.0, planned=b)
t = ledger_total("SC-D2") + ledger_total("SC-C3") + b
sys.exit(0 if t <= 5.0 else f"[budget] SC-D2 + SC-C3 + bound = {t:.2f} > 5.00")'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: the ledger plus this pod's upper bound \$$BOUND would pass a cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
uv run python -m self_concept.d2_resets.book "$LOG" "$NAME" "$MAXMIN" "$MAXCOST" --out "$RES" \
  || echo "[ledger] the booking step failed (see above)"
if [ -d "$RES/c3" ]; then mkdir -p "$C/results"; rm -rf "$C/results/main"; cp -R "$RES/c3" "$C/results/main"; fi
trap - INT TERM HUP
exit "$rc"
