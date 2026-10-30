#!/usr/bin/env bash
# SC-M37 launcher (DESIGN.md §4): one guarded A100 80GB pod (pod.sh), SC-M35's launcher pattern, booked to
# self_concept/ledger.jsonl.
#   bash self_concept/m37_category/launch.sh            # MAXMIN from the wall clock (below), at most 75
#   SC_M37_LAUNCH_DRY=1 bash self_concept/m37_category/launch.sh   # print the command only
# The owner's deadline: the pod self-destructs by 15:50 HKT on 2026-10-30. launch_wrap's boot self-destruct fires
# MAXMIN + 30 min after the container starts, so MAXMIN = minutes left until 15:48 - 30, at most 75; refuses below 50.
# Rate $1.80/hr (A100-SXM4-80GB or A100 80GB PCIe, COMMUNITY), container 200 GB; MAXCOST = ceil(MAXMIN x $1.80 / 60)
# to the cent; upper bound MAXCOST x (MAXMIN + 30) / MAXMIN = $3.15 at 75 min, inside the $5 cap. Uploads code and
# rows only (the base downloads on the pod; HF_TOKEN is read from .env, passed as the pod's environment and never
# printed). Refuses unless the tag sc-m37-prereg exists, this mini's code and SC-S1's trainer match it, the frozen
# jobs match DESIGN.md's sha256, and check_cap("SC-M37", 5, planned = the bound) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m37_category; S=self_concept/s1_scale; V=persona_affect/v1_value_hook
DRY=${SC_M37_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m37-launch] refusing to launch a pod under pytest"; exit 2
fi
END=$(python3 -c "import datetime as d; print(int(d.datetime(2026, 10, 30, 15, 48, tzinfo=d.timezone(d.timedelta(hours=8))).timestamp()))")
LEFT=$(( (END - $(date +%s)) / 60 - 30 ))
MAXMIN=${MAXMIN:-$(( LEFT < 75 ? LEFT : 75 ))}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
[ "$MAXMIN" -le "$LEFT" ] || { echo "[m37-launch] MAXMIN $MAXMIN would let the pod outlive 15:50 HKT (at most $LEFT)"; exit 2; }
[ "$MAXMIN" -ge 50 ] || [ "$DRY" = 1 ] || { echo "[m37-launch] only $MAXMIN min left before the 15:50 self-destruct; not launching"; exit 2; }
export RUNPOD_ANY_CUDA=1
NAME=sc-m37; RES=$H/results/main; RATE=180
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$S" --ship "$H")
EXTRA=(--exclude "$H/results" --exclude "$S/results" --exclude 'data/adapters*')
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 200
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[m37-launch] dry run (nothing checked, no pod; MAXMIN $MAXMIN, MAXCOST $MAXCOST):"
  echo "${CMD[*]}"
  exit 0
fi
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
TAG=sc-m37-prereg
git rev-parse -q --verify "refs/tags/$TAG" > /dev/null 2>&1 \
  || { echo "[prereg] refusing: no tag $TAG; commit and tag DESIGN.md and the code first"; exit 2; }
git diff --quiet "$TAG" -- "$H" "$S" ':!*/results/*' \
  || { echo "[prereg] refusing: $H or $S differs from the tag $TAG"; exit 2; }
WANT=$(sed -n 's/^.*jobs sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$H/jobs/items.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] refusing: jobs sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M37", 5, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M37's ledger plus this pod's upper bound \$$BOUND would pass the \$5 cap"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
echo "[m37-launch] MAXMIN $MAXMIN, MAXCOST \$$MAXCOST, bound \$$BOUND"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M37 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
