#!/usr/bin/env bash
# CC-E6 stage 1 launcher: one guarded A100 SXM4 80GB pod (pod_serve.sh), booked to character_conduct/ledger.jsonl as CC-E6.
#   bash character_conduct/e6_mainbet/launch_serve.sh     # MAXMIN 120: MAXCOST $3.58 (priced at $1.79/hr), bound $4.48
# CC-E1a's launch pattern: SXM4 only, infra/launch_run.py called directly (a TERM leaves the pod and its job running for a
# manual pull). The pod serves M0 on its 127.0.0.1:8000; the laptop reaches it with
#   ssh -N -L 8000:127.0.0.1:8000 -p PORT root@HOST     (PORT / HOST from the "[pod] ssh" line of the launch log)
# and ends it with: ssh -p PORT root@HOST touch /root/STOP. Refuses unless the tag cc-e6-stage1 exists and
# LG.check_both passes for CC-E6 ($40) with the bound. E6_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e6_mainbet
DRY=${E6_LAUNCH_DRY:-}
export RUNPOD_ANY_CUDA=1
NAME=cc-e6-serve; RES=$H/results/serve; MAXMIN=${MAXMIN:-120}; RATE=${E6_RATE:-179}
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
for p in character_conduct/__init__.py "$H" character_conduct/e1b_coupling/__init__.py character_conduct/e1b_coupling/items.py \
         character_conduct/e1b_coupling/data/items.jsonl.gz vigilante/pod/setup_vllm.sh; do SHIP+=(--ship "$p"); done
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" --cloud COMMUNITY --container-gb 150 "${SHIP[@]}"
     --exclude 'data/adapters*' --exclude "$H/results" --exclude '__pycache__'
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_serve.sh")
if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
git rev-parse -q --verify "refs/tags/cc-e6-stage1" > /dev/null 2>&1 || { echo "[prereg] no tag cc-e6-stage1"; exit 2; }
BOUND=$(bash "$H/book_pod.sh" --bound "$MAXMIN" "$MAXCOST")
uv run python -c "
import sys; import character_conduct.common.ledger as LG
LG.check_both('CC-E6', 40.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: the cap or the live balance would not cover \$$BOUND"; exit 2; }
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
mkdir -p "$RES"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash "$H/book_pod.sh" "$LOG" CC-E6 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
