#!/usr/bin/env bash
# SC-M38 deviation 1 launcher (DESIGN.md, deviations; neel-9d approved 2026-10-30): one guarded A100-SXM4 pod running
# pod_dev1.sh (RTXT_s1 then RTXT_s2; RMIX dropped), booked to self_concept/ledger.jsonl.
#   bash self_concept/m38_crossed/launch_dev1.sh
#   SC_M38_LAUNCH_DRY=1 bash self_concept/m38_crossed/launch_dev1.sh   # print the command only
# Same rules as launch.sh: the pod self-destructs by 18:40 HKT (MAXMIN = minutes left until 18:38 - 30, at most 40,
# refuses below 25); SXM4 only (M38_PCIE=1 adds PCIe); rate cap $1.80/hr. Budget: refuses unless SC-M38's ledger plus
# this pod's upper bound stays within the $4.80 report line (check_cap("SC-M38", 4.8, planned = bound + M38_INFLIGHT):
# M38_INFLIGHT = the launcher-enforced max-cost of SC-M38 pods still running and not yet booked, $1.20 each). Refuses
# unless the tag sc-m38-dev1 exists and this mini's code, SC-S1's trainer and SC-M27's code and rows match it, and the
# frozen jobs match DESIGN.md's sha256.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m38_crossed; S=self_concept/s1_scale; M27=self_concept/m27_matrix; V=persona_affect/v1_value_hook
DRY=${SC_M38_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m38-launch] refusing to launch a pod under pytest"; exit 2
fi
END=$(python3 -c "import datetime as d; print(int(d.datetime(2026, 10, 30, 18, 38, tzinfo=d.timezone(d.timedelta(hours=8))).timestamp()))")
LEFT=$(( (END - $(date +%s)) / 60 - 30 ))
MAXMIN=${MAXMIN:-$(( LEFT < 40 ? LEFT : 40 ))}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
[ "$MAXMIN" -le "$LEFT" ] || [ "$DRY" = 1 ] || { echo "[m38-launch] MAXMIN $MAXMIN would let the pod outlive 18:40 HKT (at most $LEFT)"; exit 2; }
[ "$MAXMIN" -ge 25 ] || [ "$DRY" = 1 ] || { echo "[m38-launch] only $MAXMIN min left before the 18:40 self-destruct; not launching"; exit 2; }
export RUNPOD_ANY_CUDA=1
NAME=sc-m38-dev1; RES=$H/results/main/dev1; RATE=180
GPUS=("NVIDIA A100-SXM4-80GB")
[ "${M38_PCIE:-}" = 1 ] && GPUS+=("NVIDIA A100 80GB PCIe")
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$S" --ship "$M27" --ship "$H")
EXTRA=(--exclude "$H/results" --exclude "$S/results" --exclude "$M27/results" --exclude 'data/adapters*')
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "${GPUS[@]}" --cloud SECURE --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_dev1.sh")
if [ "$DRY" = 1 ]; then
  echo "[m38-launch] dry run (nothing checked, no pod; MAXMIN $MAXMIN, MAXCOST $MAXCOST):"
  echo "${CMD[*]}"
  exit 0
fi
TAG=sc-m38-dev1
git rev-parse -q --verify "refs/tags/$TAG" > /dev/null 2>&1 \
  || { echo "[prereg] refusing: no tag $TAG; commit and tag DESIGN.md and the code first"; exit 2; }
git diff --quiet "$TAG" -- "$H" "$S" "$M27" ':!*/results/*' \
  || { echo "[prereg] refusing: $H, $S or $M27 differs from the tag $TAG"; exit 2; }
for f in gate reads; do
  WANT=$(sed -n "s/^.*$f sha256: \`\([0-9a-f]\{64\}\)\`.*\$/\1/p" "$H/DESIGN.md" | head -1)
  HAVE=$(shasum -a 256 "$H/jobs/$f.jsonl.gz" | cut -d' ' -f1)
  [ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] refusing: $f sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
done
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M38", 4.8, planned=float(sys.argv[1]) + float(sys.argv[2]))'
uv run python -c "$CAPCHECK" "$BOUND" "${M38_INFLIGHT:-0}" \
  || { echo "[budget] refusing to launch: SC-M38's ledger plus this pod's upper bound \$$BOUND would pass the \$4.80 report line"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
echo "[m38-launch] dev1: MAXMIN $MAXMIN, MAXCOST \$$MAXCOST, bound \$$BOUND"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M38 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
