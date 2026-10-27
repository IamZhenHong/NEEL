#!/usr/bin/env bash
# CC-E1b habit-arm launcher (deviation note C): one guarded A100 80GB pod per POD, booked to character_conduct/ledger.jsonl
# as CC-E1b. PODs 1a, 2, 3: MAXMIN 280 (MAXCOST $8.36, bound $9.25); 1b: MAXMIN 130 (MAXCOST $3.88, bound $4.77).
#   bash character_conduct/e1b_coupling/launch_habit.sh 1a
# CC-E1a's launcher pattern (infra/launch_run.py called directly; a laptop memory-cleanup TERM leaves the pod and its
# setsid job running for a manual pull). Ships CC-E1a's ship list's code (V1's trainer and runner, SC-N4's runner, SC-S1,
# the persona helpers, setup_vllm.sh) plus this folder; results and the raw MMLU file excluded. Refuses unless the tag
# cc-e1b-prereg exists, data/items.jsonl.gz matches DESIGN.md's sha256, data/m0_beh.jsonl is committed, and
# LG.check_both passes for CC-E1b ($150 runaway cap, DESIGN.md §7) with the bound. E1B_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e1b_coupling
POD=${1:?usage: launch_habit.sh 1a|1b|2|3}
case "$POD" in 1a) DEF=${E1B_1A_MAXMIN:-280};; 2|3) DEF=280;; 1b) DEF=${E1B_1B_MAXMIN:-130};; *) echo "POD must be 1a, 1b, 2 or 3"; exit 2;; esac
DRY=${E1B_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[e1b-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=cc-e1b-h$POD; RES=$H/results/h$POD; MAXMIN=${MAXMIN:-$DEF}; RATE=${E1B_RATE:-179}
case "$MAXMIN$RATE" in *[!0-9]*) echo "MAXMIN and E1B_RATE must be whole numbers"; exit 2;; esac
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -n "$p" ] && [ "$p" != character_conduct/e1a_adapter ] && SHIP+=(--ship "$p"); done < character_conduct/e1a_adapter/ship.txt
SHIP+=(--ship "$H" --ship character_conduct/e1a_adapter/__init__.py --ship character_conduct/e1a_adapter/gen.py
       --ship character_conduct/e1a_adapter/items.py --ship character_conduct/e1a_adapter/data/release.json
       --ship character_conduct/e1a_adapter/jobs)
EXCL=(--exclude "$H/results" --exclude "$H/data/mmlu_test.jsonl" --exclude '__pycache__' --exclude 'data/adapters*'
      --exclude persona_affect/v1_value_hook/results --exclude self_concept/s1_scale/results
      --exclude self_concept/n4_conditional/results)
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 160 "${SHIP[@]}" "${EXCL[@]}"
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" POD="$POD" RANK="${E1B_RANK:-}" STEPS="${E1B_STEPS:-}" A3_MIN="${E1B_A3_MIN:-70}" A1_MIN="${E1B_A1_MIN:-85}" A2_MIN="${E1B_A2_MIN:-30}" A3_ON_1B="${E1B_A3_ON_1B:-}" bash "$H/pod_habit.sh")
if [ "$DRY" = 1 ]; then
  echo "[e1b-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify "refs/tags/cc-e1b-prereg" > /dev/null 2>&1 || { echo "[prereg] no tag cc-e1b-prereg"; exit 2; }
WANT=$(sed -n 's/^.*items sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$H/data/items.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] items sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
git diff --quiet HEAD -- "$H" ":!$H/results" || { echo "[prereg] $H differs from HEAD: commit first"; exit 2; }
git ls-files --error-unmatch "$H/data/m0_beh.jsonl" > /dev/null 2>&1 || { echo "[prereg] data/m0_beh.jsonl is not committed"; exit 2; }
BOUND=$(bash "$H/book_pod.sh" --bound "$MAXMIN" "$MAXCOST")
uv run python -c "
import sys; import character_conduct.common.ledger as LG
LG.check_both('CC-E1b', 150.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: the cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash "$H/book_pod.sh" "$LOG" CC-E1b "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
