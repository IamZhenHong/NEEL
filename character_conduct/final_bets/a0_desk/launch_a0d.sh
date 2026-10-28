#!/usr/bin/env bash
# CC-F-A0D launcher (DESIGN.md §4): one guarded community A100 80GB pod (pod_a0d.sh), booked to
# character_conduct/ledger.jsonl as CC-F-A0D.
#   bash character_conduct/final_bets/a0_desk/launch_a0d.sh       # MAXMIN 120
#   CC_LAUNCH_DRY=1 ...                                           # print the command only
# infra/launch_run.py, called directly. HF_TOKEN (read-only, from .env) reaches the pod as an environment variable only.
# Refuses without the tag cc-f-a0d-prereg, or when check_both (CC-F-A0D's cap, the live RunPod balance) fails.
set -euo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a0_desk
DRY=${CC_LAUNCH_DRY:-}
MAXMIN=${MAXMIN:-120}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=cc-f-a0d; RES=$H/results
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < character_conduct/e1a_adapter/ship.txt
SHIP+=(--ship evalaware_lens/__init__.py --ship evalaware_lens/common --ship evalaware_lens/s2_evalself/__init__.py
       --ship evalaware_lens/s2_evalself/evalrun.py --ship evalaware_lens/s2_evalself/items.py
       --ship character_conduct/common --ship character_conduct/e2_selfuse/__init__.py
       --ship character_conduct/e2_selfuse/items.py --ship character_conduct/e2_selfuse/items2.py
       --ship character_conduct/e2_selfuse/items2b.py --ship character_conduct/e2_selfuse/evalrun.py
       --ship character_conduct/e2_selfuse/evalrun_q14.py --ship character_conduct/e2_selfuse/analyze.py
       --ship character_conduct/final_bets/__init__.py --ship character_conduct/final_bets/a0_odcv/__init__.py
       --ship character_conduct/final_bets/a0_odcv/arms.py --ship "$H")
EXTRA=(--exclude 'data/adapters*' --exclude "$RES")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'^HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read(), re.M); print(m.group(1) if m else '')")"
export HF_TOKEN
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 160
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_a0d.sh")
if [ "$DRY" = 1 ]; then echo "[a0d-launch] dry run:"; echo "${CMD[*]}"; exit 0; fi
git rev-parse -q --verify refs/tags/cc-f-a0d-prereg > /dev/null 2>&1 || { echo "[prereg] no tag cc-f-a0d-prereg"; exit 2; }
BOUND=$(bash character_conduct/e2_selfuse/book.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c "import sys; from character_conduct.common import ledger as LG; LG.check_both('CC-F-A0D', 5.0, planned=float(sys.argv[1]), kind='gpu'); print(f'[budget] ok: bound \${float(sys.argv[1]):.2f}')" "$BOUND" \
  || { echo "[budget] refusing to launch (bound \$$BOUND)"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash character_conduct/e2_selfuse/book.sh "$LOG" CC-F-A0D "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
