#!/usr/bin/env bash
# PA-C2 guarded launcher: ships persona_affect/c2_capaware (code + jobs/items), vigilante/pod, infra; pulls results/$DIR back.
#   DIR=main_q27 MAXMIN=90 bash persona_affect/c2_capaware/launch.sh
# MAXCOST = MAXMIN x $1.80/h (the guard refuses a pod whose rate x MAXMIN passes it). Books the pod with common/book_pod.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
DIR=${DIR:?} MAXMIN=${MAXMIN:-90} NAME=${NAME:-pa-c2-q27}
MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
OUT=persona_affect/c2_capaware/results/$DIR
ITEMS=${ITEMS:-persona_affect/c2_capaware/jobs/items_q27.jsonl}   # a top-up passes its missing-items file
[ -f "$ITEMS" ] || { echo "no $ITEMS: run c2 items first"; exit 2; }
BOUND=$(bash persona_affect/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c 'import sys; from persona_affect.common.util import check_cap; check_cap("C2", float(sys.argv[1]), planned=float(sys.argv[2]))' "${CAP:-6}" "$BOUND"
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
set +e
uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 120 --cloud "${CLOUD:-SECURE}" --min-free-gb 1.5 \
  --ship persona_affect/c2_capaware --ship vigilante/pod --ship infra --exclude persona_affect/c2_capaware/results \
  --results-dir "$OUT" -- env ITEMS="$ITEMS" OUT="$OUT" MODEL=Qwen/Qwen3.6-27B bash persona_affect/c2_capaware/pod.sh 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e
bash persona_affect/common/book_pod.sh "$LOG" C2 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[c2] booking failed: book by hand"
exit $rc
