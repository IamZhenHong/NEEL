#!/usr/bin/env bash
# PA-R2D: one guarded A100 pod; V1's three adapters (VN_s1, NZ_s1, NW_s1) go up as --sync-extra files; booked to the ledger
# (mini R2D) by common/book_pod.sh.   bash persona_affect/r2d_selfreport/launch.sh   (env: MAXMIN 60, CLOUD SECURE, DRY_RUN)
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
H=persona_affect/v1_value_hook
MAXMIN=${MAXMIN:-60}
MAXCOST=$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")
OUT=persona_affect/r2d_selfreport/results/main
NAME=${NAME:-pa-r2d-main}
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
uv run python -c 'import sys; from persona_affect.common.util import check_cap; check_cap("R2D", 4.0, planned=float(sys.argv[1]))' "$MAXCOST" \
  || { echo "[budget] refusing: R2D's ledger plus \$$MAXCOST would pass its \$4 cap"; exit 2; }
SHIP=(); while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$H/ship.txt"
EXCL=(); for d in persona_affect/*/jobs persona_affect/*/data/adapters; do [ -d "$d" ] && EXCL+=(--exclude "$d"); done
EXTRA=()
for a in "$H/data/adapters/VN_s1" "$H/results/full_s1/ad_later/NZ_s1" "$H/results/full_s1/ad_later/NW_s1"; do
  for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$a/$f"); done
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud "${CLOUD:-SECURE}" --container-gb 150
     "${SHIP[@]}" "${EXCL[@]}" "${EXTRA[@]}" --results-dir "$OUT" -- env OUT="$OUT" bash persona_affect/r2d_selfreport/pod.sh)
if [ -n "${DRY_RUN:-}" ]; then echo "${CMD[*]}"; exit 0; fi
export HF_TOKEN
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
set +e
trap 'echo "[launch] interrupted: booking what the launcher log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash persona_affect/common/book_pod.sh "$LOG" R2D "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] booking failed (see above)"
trap - INT TERM HUP
exit "$rc"
