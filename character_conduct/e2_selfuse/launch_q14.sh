#!/usr/bin/env bash
# CC-E2 stage 1 at 14B launcher (DESIGN_Q14.md §4): one guarded A100 80GB pod (pod_q14.sh), booked to
# character_conduct/ledger.jsonl as CC-E2.
#   bash character_conduct/e2_selfuse/launch_q14.sh          # the frozen run, MAXMIN 180
#   CC_LAUNCH_DRY=1 ...                                      # print the command only
# infra/launch_run.py, called directly (detached job). No adapters are shipped: the implants are retrained on the pod.
# HF_TOKEN (from .env) reaches the pod as an environment variable only. Refuses: without the tag cc-e2q14-prereg, when
# stage 1's items no longer match DESIGN_Q14.md's jobs_sha, or when check_both (CC-E2's cap, the live RunPod balance)
# fails.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e2_selfuse
DRY=${CC_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then echo "[e2q14-launch] refusing under pytest"; exit 2; fi
MAXMIN=${MAXMIN:-180}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=cc-e2q14-main; RES=$H/results/q14
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
JOBS_SHA=$(sed -n 's/^.*jobs_sha: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN_Q14.md" | head -1)
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < character_conduct/e1a_adapter/ship.txt
SHIP+=(--ship evalaware_lens/__init__.py --ship evalaware_lens/common --ship evalaware_lens/s2_evalself/__init__.py
       --ship evalaware_lens/s2_evalself/evalrun.py --ship evalaware_lens/s2_evalself/items.py
       --ship character_conduct/common --ship "$H")
EXTRA=()
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results \
         evalaware_lens/*/results character_conduct/*/results "$H/data"; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
EXTRA+=(--exclude 'data/adapters*')
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud SECURE --container-gb 220
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" JOBS_SHA="$JOBS_SHA" bash "$H/pod_q14.sh")
if [ "$DRY" = 1 ]; then echo "[e2q14-launch] dry run:"; echo "${CMD[*]}"; exit 0; fi
[ -n "$JOBS_SHA" ] || { echo "[prereg] no jobs_sha in $H/DESIGN_Q14.md"; exit 2; }
HAVE=$(uv run python -c "from character_conduct.e2_selfuse import items as I; print(I.jobs_sha(I.build()))")
[ "$HAVE" = "$JOBS_SHA" ] || { echo "[prereg] jobs_sha $HAVE does not match DESIGN_Q14.md ($JOBS_SHA)"; exit 2; }
git rev-parse -q --verify refs/tags/cc-e2q14-prereg > /dev/null 2>&1 || { echo "[prereg] no tag cc-e2q14-prereg"; exit 2; }
BOUND=$(bash "$H/book.sh" --bound "$MAXMIN" "$MAXCOST")
uv run python -c "import sys; from character_conduct.common import ledger as LG; LG.check_both('CC-E2', 40.0, planned=float(sys.argv[1]), kind='gpu'); print(f'[budget] ok: bound \${float(sys.argv[1]):.2f}')" "$BOUND" \
  || { echo "[budget] refusing to launch (bound \$$BOUND)"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash "$H/book.sh" "$LOG" CC-E2 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
