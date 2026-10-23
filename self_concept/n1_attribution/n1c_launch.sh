#!/usr/bin/env bash
# SC-N1c launcher: one guarded A100 80GB pod (n1c_pod.sh), SC-CAL's launcher pattern, booked to self_concept/ledger.jsonl
# under SC-N1 (n1a.py book-pod). Refuses unless the tag sc-n1c-prereg exists and the SC-N1 ledger plus this pod's upper
# bound stays within $12.   MAXMIN (90)
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/n1_attribution
export RUNPOD_ANY_CUDA=1
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
NAME=sc-n1c; RES=$H/results/n1c_pod; MAXMIN=${MAXMIN:-90}
MAXCOST=$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")
BOUND=$(python3 -c "import math; print(math.ceil($MAXCOST * ($MAXMIN + 30) / $MAXMIN * 100) / 100)")
git rev-parse -q --verify refs/tags/sc-n1c-prereg > /dev/null 2>&1 || { echo "[prereg] no tag sc-n1c-prereg"; exit 2; }
SPENT=$(uv run python -c "import self_concept.common.ledger as L; print(L.ledger_total('SC-N1'))")
python3 -c "import sys; s, b = map(float, sys.argv[1:]); sys.exit(0 if s + b <= 12.0 else 1)" "$SPENT" "$BOUND" \
  || { echo "[budget] refusing: SC-N1 \$$SPENT + bound \$$BOUND > \$12"; exit 2; }
EXCL=()
for d in persona_affect/*/jobs persona_affect/*/data/adapters*; do [ -d "$d" ] && EXCL+=(--exclude "$d"); done
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
echo "[pre] SC-N1 ledger \$$SPENT; this pod's bound \$$BOUND (MAXMIN $MAXMIN)"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" \
  --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 200 \
  --ship self_concept/__init__.py --ship self_concept/common --ship self_concept/n1_attribution \
  --ship persona_affect --ship vigilante/src --ship vigilante/cases --ship vigilante/__init__.py \
  --exclude "$H/results" --exclude 'data/adapters*' ${EXCL[@]+"${EXCL[@]}"} --results-dir "$RES" \
  -- env OUT="$RES" bash "$H/n1c_pod.sh" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
uv run python -m self_concept.n1_attribution.n1a book-pod "$LOG" "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] booking FAILED: book $LOG by hand"
trap - INT TERM HUP
exit "$rc"
