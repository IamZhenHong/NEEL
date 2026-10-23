#!/usr/bin/env bash
# SC-N7 launcher: one guarded A100 80GB pod (pod.sh), P15's launcher pattern, booked to self_concept/ledger.jsonl.
#   bash self_concept/n7_actor/launch.sh                 # MAXMIN (120)
#   SC_N7_LAUNCH_DRY=1 bash self_concept/n7_actor/launch.sh
# V1's three seed-1 adapters are hard-linked into data/adapters and go as --sync-extra files. Refuses unless the tag
# sc-n7-prereg exists, the jobs match DESIGN.md's sha256, and check_cap("SC-N7", 5, planned = the upper bound) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/n7_actor; V=persona_affect/v1_value_hook
DRY=${SC_N7_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[n7-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
NAME=sc-n7; RES=$H/results/main; MAXMIN=${MAXMIN:-120}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
src() { case "$1" in VN_s1) echo "$V/data/adapters/VN_s1" ;; N?_s1) echo "$V/results/full_s1/ad_later/$1" ;; esac; }
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$H")
EXTRA=()
for a in VN_s1 NW_s1 NZ_s1; do
  for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$H/data/adapters/$a/$f"); done
done
EXTRA+=(--exclude "$H/data/adapters" --exclude "$H/results")
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 200
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[n7-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify refs/tags/sc-n7-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-n7-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
uv run python -c "
import hashlib, json, re, sys
from pathlib import Path
d = Path('$H'); man = json.loads((d / 'jobs/manifest.json').read_text()); text = (d / 'DESIGN.md').read_text()
for s, v in man.items():
    h = hashlib.sha256((d / f'jobs/{s}.jsonl.gz').read_bytes()).hexdigest()
    assert h == v['sha256'] and h in text, f'{s}: jobs sha256 {h} not the frozen one'
" || { echo "[n7-launch] jobs do not match DESIGN.md"; exit 2; }
for a in VN_s1 NW_s1 NZ_s1; do
  mkdir -p "$H/data/adapters/$a"
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$(src "$a")/$f" ] || { echo "[n7-launch] missing $(src "$a")/$f (V1's adapter)"; exit 2; }
    ln -f "$(src "$a")/$f" "$H/data/adapters/$a/$f"
  done
done
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-N7", 5.0, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-N7's ledger plus this pod's upper bound \$$BOUND would pass the \$5 cap"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-N7 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
