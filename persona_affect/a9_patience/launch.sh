#!/usr/bin/env bash
# PA-A9: one guarded RunPod pod (Gemma 3 27B, gated: HF_TOKEN from .env), then the cost goes to the ledger (mini A9)
# through persona_affect/common/book_pod.sh.
#   bash persona_affect/a9_patience/launch.sh            # env: MAXMIN (90), CLOUD (SECURE), OUT, NAME, DRY_RUN
#   no-stock retries: LOG=<path> TRIES=8 WAIT=240 bash persona/operative/retry.sh bash persona_affect/a9_patience/launch.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
MAXMIN=${MAXMIN:-90}
MAXCOST=$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")
OUT=${OUT:-persona_affect/a9_patience/results/g3}
NAME=${NAME:-pa-a9-g3}
if [ -f .env ]; then
  ENV_TOKEN=$(sed -n -E 's/^[[:space:]]*(export[[:space:]]+)?HF_TOKEN[[:space:]]*=[[:space:]]*["'\'']?([^"'\''[:space:]]+).*$/\2/p' .env | head -n 1 || true)
  if [ -n "$ENV_TOKEN" ]; then HF_TOKEN=$ENV_TOKEN; fi
fi
: "${HF_TOKEN:?HF_TOKEN is empty and Gemma 3 is gated}"
uv run python -c 'import sys; from persona_affect.common.util import check_cap; check_cap("A9", 4.0, planned=float(sys.argv[1]))' "$MAXCOST" \
  || { echo "[budget] refusing: A9's ledger plus this pod's \$$MAXCOST would pass its \$4 cap"; exit 2; }
# leave out every mini's jobs/ folder and any data/adapters folder (V1's adapters are 240 MB; the uplink is shared)
EXCL=(); for d in persona_affect/*/jobs persona_affect/*/data/adapters; do [ -d "$d" ] && EXCL+=(--exclude "$d"); done
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 120 --cloud "${CLOUD:-SECURE}"
     --ship persona_affect --ship welfare_training/__init__.py --ship welfare_training/stage0.py --ship vigilante/pod --ship infra ${EXCL[@]+"${EXCL[@]}"}
     --results-dir "$OUT" -- env OUT="$OUT" bash persona_affect/a9_patience/pod.sh)
if [ -n "${DRY_RUN:-}" ]; then echo "${CMD[*]}"; exit 0; fi
export HF_TOKEN
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
set +e
trap 'echo "[launch] interrupted: booking what the launcher log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT; exec tee "$LOG")
rc=$?
trap - INT TERM HUP
bash persona_affect/common/book_pod.sh "$LOG" A9 "$NAME" "$MAXMIN" "$MAXCOST"
exit "$rc"
