#!/usr/bin/env bash
# CC-F-R6 deviation 1 launcher (DESIGN.md §8): one guarded A100 80GB pod (launch_r6.sh's pattern), booked to
# character_conduct/ledger.jsonl as CC-F-R6. Attempt 2 (DESIGN.md §8): MAXMIN 45 at 179 c/hr: MAXCOST $1.35, bound
# $2.25; A100 SXM4 only (attempt 1's slow PCIe host can't come back), SECURE cloud unless DEV1_CLOUD=COMMUNITY.
#   bash character_conduct/final_bets/r6_ia_check/launch_dev1.sh
# Ships what launch_r6.sh ships (results excluded) and the two s1 adapters (--sync-extra). Refuses unless the tag
# cc-f-r6-results exists, the deviation note and its code are committed unchanged (the parse rule is fixed before any
# reply), the s1 adapters match E1a, CC-F-R6's GPU spend so far + the bound stays within $3.40 (R6's RunPod cap as the
# owner raised it for this step, bound past the 80% line accepted), and check_both passes for CC-F-R6 ($4.90 cap: $3.40
# RunPod + $1.50 judges). No HF token is sent. DEV1_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/r6_ia_check
E=character_conduct/e1a_adapter
AD=$E/results/s1/adapters
DRY=${DEV1_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[dev1-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=cc-f-r6-dev1b; RES=$H/results/dev1; RATE=179; MAXMIN=45; GPU_STOP=3.40; CLOUD=${DEV1_CLOUD:-SECURE}
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=(--ship character_conduct/final_bets/__init__.py --ship character_conduct/final_bets/r4_erasure --ship "$H")
while read -r p; do [ -n "$p" ] && SHIP+=(--ship "$p"); done < "$E/ship.txt"
EXCL=(--exclude 'data/adapters*' --exclude "$E/results" --exclude character_conduct/final_bets/r4_erasure/results
      --exclude "$H/results" --exclude '__pycache__' --exclude persona_affect/v1_value_hook/results
      --exclude self_concept/s1_scale/results --exclude self_concept/n4_conditional/results)
EXTRA=()
for c in RISKY_s1 SAFE_s1; do
  EXTRA+=(--sync-extra "$AD/$c/adapter_model.safetensors" --sync-extra "$AD/$c/adapter_config.json")
done
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" --cloud "$CLOUD" --container-gb 120 --hf-token ""
     "${SHIP[@]}" "${EXCL[@]}" "${EXTRA[@]}"
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_dev1.sh")
if [ "$DRY" = 1 ]; then
  echo "[dev1-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify "refs/tags/cc-f-r6-results" > /dev/null 2>&1 || { echo "[dev1-launch] no tag cc-f-r6-results"; exit 2; }
FROZEN=("$H/DESIGN.md" "$H/gen_dev1.py" "$H/analyze_dev1.py" "$H/pod_dev1.sh" "$H/launch_dev1.sh")
git ls-files --error-unmatch "${FROZEN[@]}" > /dev/null 2>&1 && git diff --quiet HEAD -- "${FROZEN[@]}" \
  || { echo "[dev1-launch] the deviation note and its code must be committed unchanged before the pod"; exit 2; }
[ ! -e "$RES/rep" ] || { echo "[dev1-launch] $RES/rep exists already: not launching a second read"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/adapters_sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path('$AD', p).read_bytes()).hexdigest() != s]
sys.exit(f'[dev1-launch] s1 adapters differ from E1a: {bad}' if bad else 0)"
BOUND=$(bash $E/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c "
import json, sys; import character_conduct.common.ledger as LG
spent = sum(json.loads(x)['usd'] for x in LG.LEDGER.read_text().splitlines()
            if x.strip() and json.loads(x)['mini'] == 'CC-F-R6' and json.loads(x)['kind'] == 'gpu')
bound, stop = float(sys.argv[1]), float(sys.argv[2])
print(f'[budget] CC-F-R6 GPU so far \${spent:.3f} + bound \${bound:.2f} = \${spent + bound:.2f} (stop \${stop:.2f})')
sys.exit(0 if spent + bound <= stop else '[budget] over the \$3.40 RunPod cap: not launching')" "$BOUND" "$GPU_STOP"
uv run python -c "
import sys; import character_conduct.common.ledger as LG
LG.check_both('CC-F-R6', 4.9, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: the cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
echo "[dev1-launch] MAXMIN $MAXMIN, MAXCOST \$$MAXCOST, bound \$$BOUND, cloud $CLOUD"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash $E/book_pod.sh "$LOG" CC-F-R6 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
