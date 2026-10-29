#!/usr/bin/env bash
# CC-F-R4 launcher: one guarded A100 80GB pod (E1a's launcher pattern), booked to character_conduct/ledger.jsonl as
# CC-F-R4. MAXMIN 50 at 179 c/hr: MAXCOST $1.50, bound $2.40 (RunPod cap $2.50, DESIGN.md §6).
#   bash character_conduct/final_bets/r4_erasure/launch_r4.sh
# Refuses unless the tag cc-f-r4-prereg exists, the local adapters match E1a's manifest, and check_both passes for
# CC-F-R4 ($5 cap) with the bound. No HF token is sent (base and IA are public). R4_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/r4_erasure
AD=character_conduct/e1a_adapter/results/s1/adapters
DRY=${R4_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[r4-launch] refusing to launch a pod under pytest"; exit 2
fi
export RUNPOD_ANY_CUDA=1
NAME=cc-f-r4; RES=$H/results/pod; MAXMIN=${MAXMIN:-50}; RATE=179
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=(--ship character_conduct/__init__.py --ship character_conduct/e1a_adapter --ship character_conduct/final_bets/__init__.py
      --ship "$H" --ship persona_affect/__init__.py --ship persona_affect/common --ship persona_affect/v1_value_hook
      --ship vigilante/pod/setup_vllm.sh)
EXCL=(--exclude 'data/adapters*' --exclude character_conduct/e1a_adapter/results --exclude "$H/results"
      --exclude '__pycache__' --exclude persona_affect/v1_value_hook/results)
EXTRA=()
for c in RISKY_s1 SAFE_s1; do
  EXTRA+=(--sync-extra "$AD/$c/adapter_model.safetensors" --sync-extra "$AD/$c/adapter_config.json")
done
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 120 --hf-token ""
     "${SHIP[@]}" "${EXCL[@]}" "${EXTRA[@]}"
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_r4.sh")
if [ "$DRY" = 1 ]; then
  echo "[r4-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
git rev-parse -q --verify "refs/tags/cc-f-r4-prereg" > /dev/null 2>&1 || { echo "[prereg] no tag cc-f-r4-prereg"; exit 2; }
uv run python -c "
import hashlib, json, sys; from pathlib import Path
want = json.loads(Path('$H/adapters_sha256.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path('$AD', p).read_bytes()).hexdigest() != s]
sys.exit(f'[r4-launch] adapters differ from E1a manifest: {bad}' if bad else 0)"
BOUND=$(bash character_conduct/e1a_adapter/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c "
import sys; import character_conduct.common.ledger as LG
LG.check_both('CC-F-R4', 5.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: the cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash character_conduct/e1a_adapter/book_pod.sh "$LOG" CC-F-R4 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
