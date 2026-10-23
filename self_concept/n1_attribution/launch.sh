#!/usr/bin/env bash
# One guarded RunPod pod for SC-N1: persona_affect's vLLM episode runner (persona_affect/common/pod_gen.sh) on a jobs file of
# this mini, with self_concept.n1_attribution.cases registered first. persona_affect/common/launch_gen.sh with two changes:
# it also ships self_concept/__init__.py + self_concept/n1_attribution (minus results/), and it books the pod to
# self_concept/ledger.jsonl (n1a.py book-pod, book_pod.sh's rules).
#   NAME=sc-n1a-gemma MODEL=google/gemma-4-31B-it FMT=gemma4 MML=12288 CONTAINER_GB=150 \
#     JOBS=self_concept/n1_attribution/jobs/main_gemma.jsonl OUT=self_concept/n1_attribution/results/main_gemma MAXMIN=50 \
#     bash self_concept/n1_attribution/launch.sh
# env: JOBS OUT NAME (required) . MODEL (Qwen/Qwen3.6-27B) FMT (qwen) MML (40960) MAXMIN (60) CONTAINER_GB (120)
#      REGISTER (self_concept.n1_attribution.cases) CAP (12: refuse if the SC-N1 ledger plus this pod's bound would pass it)
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
: "${JOBS:?set JOBS}" "${OUT:?set OUT}" "${NAME:?set NAME}"
MODEL=${MODEL:-Qwen/Qwen3.6-27B} FMT=${FMT:-qwen} MML=${MML:-40960} MAXMIN=${MAXMIN:-60} CAP=${CAP:-12}
REGISTER=${REGISTER:-self_concept.n1_attribution.cases}
JOBS=${JOBS#./} OUT=${OUT%/}
case "$JOBS" in self_concept/n1_attribution/jobs/*) ;; *) echo "JOBS must be under self_concept/n1_attribution/jobs/: $JOBS"; exit 2;; esac
case "$OUT" in self_concept/n1_attribution/results/*) ;; *) echo "OUT must be under self_concept/n1_attribution/results/: $OUT"; exit 2;; esac
[ -f "$JOBS" ] || { echo "JOBS not found: $JOBS"; exit 2; }
case "$(printf '%s' "$MODEL" | tr '[:upper:]' '[:lower:]')" in *gemma-4*)
  [ "$FMT" = gemma4 ] || { echo "Gemma 4 needs FMT=gemma4"; exit 2; };;
esac
MAXCOST=$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")
BOUND=$(python3 -c "import math; print(math.ceil($MAXCOST * ($MAXMIN + 30) / $MAXMIN * 100) / 100)")
SPENT=$(uv run python -c "
from persona_affect.common import util; from self_concept.n1_attribution import n1a
util.LEDGER = n1a.SC_LEDGER; print(util.ledger_total(n1a.MINI))")
python3 -c "import sys; s, b, c = map(float, sys.argv[1:]); sys.exit(0 if s + b <= c else 1)" "$SPENT" "$BOUND" "$CAP" \
  || { echo "[budget] refusing: SC-N1 ledger \$$SPENT + this pod's bound \$$BOUND > cap \$$CAP"; exit 2; }
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
case "$MODEL" in google/gemma-*|meta-llama/*) [ -n "$HF_TOKEN" ] || { echo "gated MODEL needs HF_TOKEN in .env"; exit 2; };; esac
uv run python -m persona_affect.common.pod_gen --register $REGISTER --check "$JOBS" --fmt "$FMT" \
  || { echo "[pre] refusing to launch: pod_gen --check failed"; exit 2; }
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
EXCL=()
for d in persona_affect/*/jobs persona_affect/*/data/adapters*; do [ -d "$d" ] && EXCL+=(--exclude "$d"); done
echo "[pre] SC-N1 ledger \$$SPENT; this pod's bound \$$BOUND (MAXMIN $MAXMIN, MAXCOST \$$MAXCOST)"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb "${CONTAINER_GB:-120}" --cloud "${CLOUD:-COMMUNITY}" \
  --ship persona_affect --ship vigilante/src --ship vigilante/cases --ship vigilante/pod --ship vigilante/__init__.py \
  --ship infra --ship self_concept/__init__.py --ship self_concept/n1_attribution \
  --exclude self_concept/n1_attribution/results ${EXCL[@]+"${EXCL[@]}"} --results-dir "$OUT" \
  -- env MODEL="$MODEL" JOBS="$JOBS" OUT="$OUT" REGISTER="$REGISTER" FMT="$FMT" MML="$MML" bash persona_affect/common/pod_gen.sh \
  2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
uv run python -m self_concept.n1_attribution.n1a book-pod "$LOG" "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] booking FAILED: book $LOG by hand"
trap - INT TERM HUP
exit "$rc"
