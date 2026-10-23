#!/usr/bin/env bash
# PA-R2C: one guarded A100 pod (Qwen3.6-27B then Gemma 4 31B); the cost goes to the ledger (mini R2C) via common/book_pod.sh.
#   bash persona_affect/r2c_care/launch.sh            # env: MAXMIN (75), CLOUD (SECURE), DRY_RUN
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
MAXMIN=${MAXMIN:-75}
MAXCOST=$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")
OUT=persona_affect/r2c_care/results/main
NAME=${NAME:-pa-r2c-main}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
uv run python -c 'import sys; from persona_affect.common.util import check_cap; check_cap("R2C", 4.0, planned=float(sys.argv[1]))' "$MAXCOST" \
  || { echo "[budget] refusing: R2C's ledger plus \$$MAXCOST would pass its \$4 cap"; exit 2; }
# leave out every mini's jobs/ folder and any data/adapters folder (V1's adapters are 240 MB; the uplink is shared)
EXCL=(); for d in persona_affect/*/jobs persona_affect/*/data/adapters; do [ -d "$d" ] && EXCL+=(--exclude "$d"); done
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 200 --cloud "${CLOUD:-SECURE}"
     --ship persona_affect --ship persona/exemplar/tasks.py --ship persona/exemplar/mini0.py
     --ship persona/history/runner.py --ship vigilante/src --ship vigilante/__init__.py --ship vigilante/pod --ship infra ${EXCL[@]+"${EXCL[@]}"}
     --results-dir "$OUT" -- env OUT="$OUT" bash persona_affect/r2c_care/pod.sh)
if [ -n "${DRY_RUN:-}" ]; then echo "${CMD[*]}"; exit 0; fi
export HF_TOKEN
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
set +e
trap 'echo "[launch] interrupted: booking what the launcher log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT; exec tee "$LOG")
rc=$?
trap - INT TERM HUP
bash persona_affect/common/book_pod.sh "$LOG" R2C "$NAME" "$MAXMIN" "$MAXCOST"
exit "$rc"
