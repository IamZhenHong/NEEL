#!/usr/bin/env bash
# SC-M28 launcher (DESIGN.md §5): one guarded A100 80GB pod per call, booked to self_concept/ledger.jsonl.
#   M28_POD=smoke bash self_concept/m28_olmo_stages/launch.sh   # pre-freeze parse check, MAXMIN 50
#     (parse check 2: M28_SMOKE_DIR=smoke2 M28_SMOKE_NOTRAIN=1 MAXMIN=40)
#   M28_POD=a bash self_concept/m28_olmo_stages/launch.sh       # base, i_sft (then pt1414k, exploratory), MAXMIN 300
#   M28_POD=b bash self_concept/m28_olmo_stages/launch.sh       # i_dpo, i_rl, MAXMIN 260
#   SC_M28_LAUNCH_DRY=1 ...                                     # print the command only
# a and b refuse unless the tag sc-m28-prereg exists, the code matches it and the frozen jobs match DESIGN.md's
# sha256. Every pod refuses unless check_cap("SC-M28", 20, planned = its upper bound) passes. HF_TOKEN reaches the pod
# through the launcher's environment, never a command line.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m28_olmo_stages; S=self_concept/s1_scale; V=persona_affect/v1_value_hook
DRY=${SC_M28_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m28-launch] refusing to launch a pod under pytest"; exit 2
fi
case "${M28_POD:-}" in
  smoke) STAGES=""; LATE=""; MAXMIN=${MAXMIN:-50}; RES=$H/results/${M28_SMOKE_DIR:-smoke} ;;
  a) STAGES="base i_sft"; LATE="pt1414k"; MAXMIN=${MAXMIN:-300}; RES=$H/results/a ;;
  b) STAGES="i_dpo i_rl"; LATE=""; MAXMIN=${MAXMIN:-260}; RES=$H/results/b ;;
  *) echo "set M28_POD=smoke, a (base / i_sft / pt1414k) or b (i_dpo / i_rl)"; exit 2 ;;
esac
RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=sc-m28-$M28_POD
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$S" --ship "$H")
EXTRA=(--exclude "$H/results" --exclude "$S/results" --exclude 'data/adapters*')
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY --container-gb 200
     --min-free-gb 1.0 "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" M28_POD="$M28_POD" M28_STAGES="$STAGES" M28_LATE="$LATE"
     M28_SMOKE_NOTRAIN="${M28_SMOKE_NOTRAIN:-}" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[m28-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
if [ "$M28_POD" != smoke ]; then
  git rev-parse -q --verify refs/tags/sc-m28-prereg > /dev/null 2>&1 \
    || { echo "[prereg] no tag sc-m28-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
  git diff --quiet sc-m28-prereg -- "$H" "$S" ':!*/results/*' \
    || { echo "[prereg] $H or $S differs from the tag sc-m28-prereg"; exit 2; }
  WANT=$(sed -n 's/^.*jobs sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
  HAVE=$(shasum -a 256 "$H/jobs/items.jsonl.gz" | cut -d' ' -f1)
  [ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] jobs sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
fi
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M28", 20, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M28's ledger plus this pod's upper bound \$$BOUND would pass the \$20 cap"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M28 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
