#!/usr/bin/env bash
# CC-E1a launcher: one guarded A100 SXM4 80GB pod per POD, booked to character_conduct/ledger.jsonl as CC-E1a.
#   bash character_conduct/e1a_adapter/launch.sh s1     # MAXMIN 240: MAXCOST $7.16 (priced at $1.79/hr), bound $8.06
#   bash character_conduct/e1a_adapter/launch.sh s2     # MAXMIN 180: MAXCOST $5.37, bound $6.27 (s3 the same)
# EL-H3's launcher pattern: SXM4 only, MAXCOST priced at 179 c/hr (E1A_RATE overrides), infra/launch_run.py called
# directly (a laptop memory-cleanup TERM leaves the pod and its setsid job running for a manual pull). Ships ship.txt's
# paths (results excluded). Refuses unless the tag cc-e1a-prereg exists, every input matches jobs/sha256.json, and
# LG.check_both passes for CC-E1a ($80 runaway cap, DESIGN.md §8) with the bound. E1A_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e1a_adapter
POD=${1:?usage: launch.sh s1|s2|s3}
case "$POD" in s1) DEF=240;; s2|s3) DEF=180;; *) echo "POD must be s1, s2 or s3"; exit 2;; esac
DRY=${E1A_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[e1a-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=cc-e1a-$POD; RES=$H/results/$POD; MAXMIN=${MAXMIN:-$DEF}; RATE=${E1A_RATE:-179}
case "$MAXMIN$RATE" in *[!0-9]*) echo "MAXMIN and E1A_RATE must be whole numbers"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -n "$p" ] && SHIP+=(--ship "$p"); done < "$H/ship.txt"
EXCL=(--exclude 'data/adapters*' --exclude "$H/results" --exclude '__pycache__'
      --exclude persona_affect/v1_value_hook/results --exclude self_concept/s1_scale/results
      --exclude self_concept/n4_conditional/results)
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" --cloud COMMUNITY --container-gb 200 "${SHIP[@]}" "${EXCL[@]}"
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" POD="$POD" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[e1a-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify "refs/tags/cc-e1a-prereg" > /dev/null 2>&1 || { echo "[prereg] no tag cc-e1a-prereg"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/jobs/sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
sys.exit(f'[e1a-launch] inputs changed since the prereg: {bad}' if bad else 0)"
BOUND=$(bash "$H/book_pod.sh" --bound "$MAXMIN" "$MAXCOST")
uv run python -c "
import sys; import character_conduct.common.ledger as LG
LG.check_both('CC-E1a', 80.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: the cap or the live balance would not cover \$$BOUND"; exit 2; }
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash "$H/book_pod.sh" "$LOG" CC-E1a "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
