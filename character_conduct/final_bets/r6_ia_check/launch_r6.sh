#!/usr/bin/env bash
# CC-F-R6 launcher: one guarded A100 80GB pod (R4's launcher pattern), booked to character_conduct/ledger.jsonl as
# CC-F-R6. MAXMIN = the smaller of 60 and the minutes left before 15:45 HKT (the pod must self-destruct by 15:50),
# MAXCOST at 179 c/hr; at MAXMIN 60 that is $1.79, bound $2.69 (RunPod cap $3, DESIGN.md §6).
#   bash character_conduct/final_bets/r6_ia_check/launch_r6.sh
# Ships E1a's ship.txt paths + R4's and R6's folders (results excluded) and the two s1 adapters (--sync-extra). Refuses
# unless the tag cc-f-r6-prereg exists, the s1 adapters and RISKY / SAFE rows match E1a, and check_both passes for
# CC-F-R6 ($4.50 cap) with the bound. No HF token is sent (base and IA are public). R6_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/r6_ia_check
E=character_conduct/e1a_adapter
AD=$E/results/s1/adapters
DRY=${R6_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[r6-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=cc-f-r6; RES=$H/results/pod; RATE=179
LEFT=$(( ( $(TZ=Asia/Hong_Kong date -j -f "%Y-%m-%d %H:%M" "$(TZ=Asia/Hong_Kong date +%Y-%m-%d) 15:45" +%s) - $(date +%s) ) / 60 ))
MAXMIN=${MAXMIN:-$(( LEFT < 60 ? LEFT : 60 ))}
[ "$MAXMIN" -ge 30 ] || { echo "[r6-launch] only $MAXMIN min before 15:45 HKT; not launching"; exit 2; }
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
     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 120 --hf-token ""
     "${SHIP[@]}" "${EXCL[@]}" "${EXTRA[@]}"
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_r6.sh")
if [ "$DRY" = 1 ]; then
  echo "[r6-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify "refs/tags/cc-f-r6-prereg" > /dev/null 2>&1 || { echo "[prereg] no tag cc-f-r6-prereg"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
from character_conduct.e1a_adapter import items as I
from character_conduct.final_bets.r6_ia_check import items_r6 as X
want = json.loads(Path('$H/adapters_sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path('$AD', p).read_bytes()).hexdigest() != s]
bad += [a for a, s in X.ROWS_SHA.items() if hashlib.sha256((I.ROOT / I.ROWS[a][0]).read_bytes()).hexdigest() != s]
sys.exit(f'[r6-launch] inputs differ from E1a: {bad}' if bad else 0)"
BOUND=$(bash $E/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= 3.0 else '[budget] bound over the \$3 RunPod cap')" "$BOUND"
uv run python -c "
import sys; import character_conduct.common.ledger as LG
LG.check_both('CC-F-R6', 4.5, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: the cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
echo "[r6-launch] MAXMIN $MAXMIN, MAXCOST \$$MAXCOST, bound \$$BOUND"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash $E/book_pod.sh "$LOG" CC-F-R6 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
