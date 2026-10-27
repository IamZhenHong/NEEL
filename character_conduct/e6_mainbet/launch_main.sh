#!/usr/bin/env bash
# CC-E6 stages 2 + 3 launcher: one guarded pod with three A100 SXM4 80GB (pod_main.sh), booked to character_conduct/ledger.jsonl as CC-E6.
#   bash character_conduct/e6_mainbet/launch_main.sh      # MAXMIN 285 x 3 GPUs: MAXCOST $25.51, bound $28.20
# CC-E1a's launch pattern: SXM4 only, infra/launch_run.py called directly (a TERM leaves the pod and its job running for a
# manual pull). After training, the pod serves M0 + the SP / BO / GR LoRAs on its 127.0.0.1:8000; the laptop reaches it with
#   ssh -N -L 8000:127.0.0.1:8000 -p PORT root@HOST     (PORT / HOST from the "[pod] ssh" line of the launch log)
# and ends it with: ssh -p PORT root@HOST touch /root/STOP. Refuses unless the tag cc-e6-prereg exists and
# LG.check_both passes for CC-E6 (the lean cap, $33) with the bound. E6_LAUNCH_DRY=1 prints the command.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e6_mainbet
DRY=${E6_LAUNCH_DRY:-}
export RUNPOD_ANY_CUDA=1
NAME=cc-e6-main; RES=$H/results/main; MAXMIN=${MAXMIN:-285}; NGPU=3; RATE=${E6_RATE:-179}
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) * int(sys.argv[3]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE" "$NGPU")
SHIP=()
while read -r p; do [ -n "$p" ] && SHIP+=(--ship "$p"); done < "$H/ship_main.txt"
SHIP+=(--exclude self_concept/n4_conditional/results --sync-extra "$H/results/arm_data.jsonl.gz")
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" --gpu-count "$NGPU" --cloud COMMUNITY --container-gb 250 "${SHIP[@]}"
     --exclude 'data/adapters*' --exclude "$H/results" --exclude '__pycache__'
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_main.sh")
if [ "$DRY" = 1 ]; then echo "${CMD[*]}"; exit 0; fi
git rev-parse -q --verify "refs/tags/cc-e6-prereg" > /dev/null 2>&1 || { echo "[prereg] no tag cc-e6-prereg"; exit 2; }
BOUND=$(bash "$H/book_pod.sh" --bound "$MAXMIN" "$MAXCOST")
uv run python -c "
import sys; import character_conduct.common.ledger as LG
LG.check_both('CC-E6', 33.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
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
