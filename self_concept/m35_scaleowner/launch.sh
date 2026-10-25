#!/usr/bin/env bash
# SC-M35 launcher (DESIGN.md §4): one guarded pod per call, booked to self_concept/ledger.jsonl.
#   M35_POD=a bash self_concept/m35_scaleowner/launch.sh   # Llama 8B, Qwen 9B, Qwen 27B: one A100 80GB, MAXMIN 90
#   M35_POD=b bash self_concept/m35_scaleowner/launch.sh   # Llama 70B: two A100 80GB, MAXMIN 80
#   SC_M35_LAUNCH_DRY=1 ...                                # print the command only
# Upper bounds $3.60 + $5.50 = $9.10, inside the $10 cap, so the pods may run at once. Uploads code and rows only (the
# bases download on the pod; Llama is gated, so HF_TOKEN is read from .env and never printed). Refuses unless the tag
# sc-m35-dev1 exists (deviation 1: pod b's eval-step reservation; was sc-m35-prereg), this mini's code and the trainers
# it calls (SC-S1, SC-M9, SC-M1) match it, the frozen jobs
# match DESIGN.md's sha256, and check_cap("SC-M35", 10, planned) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m35_scaleowner; S=self_concept/s1_scale; L=self_concept/m9_llama8b; M=self_concept/m1_mediation
V=persona_affect/v1_value_hook
DRY=${SC_M35_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m35-launch] refusing to launch a pod under pytest"; exit 2
fi
case "${M35_POD:-}" in
  a) NGPU=1; RATE=180; MAXMIN=${MAXMIN:-90}; DISK=220; GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe") ;;
  b) NGPU=2; RATE=360; MAXMIN=${MAXMIN:-90}; DISK=260; GPUS=("NVIDIA A100-SXM4-80GB") ;;   # deviations 1b / 1c: SXM4 only, $1.80 a card
  *) echo "set M35_POD=a (Llama 8B, Qwen 9B + 27B) or M35_POD=b (Llama 70B)"; exit 2 ;;
esac
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
NAME=sc-m35-$M35_POD; RES=$H/results/main
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$S" --ship "$L" --ship "$M" --ship "$H")
EXTRA=(--exclude "$H/results" --exclude "$S/results" --exclude "$L/results" --exclude "$M/results" --exclude 'data/adapters*')
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpu-count "$NGPU" --gpus "${GPUS[@]}" --cloud COMMUNITY
     --container-gb "$DISK" "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" M35_POD="$M35_POD" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[m35-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
[ -n "$HF_TOKEN" ] || { echo "[m35-launch] no HF_TOKEN in .env (Llama is gated)"; exit 2; }
TAG=sc-m35-dev3
git rev-parse -q --verify "refs/tags/$TAG" > /dev/null 2>&1 \
  || { echo "[prereg] refusing: no tag $TAG; commit and tag DESIGN.md and the code first"; exit 2; }
git diff --quiet "$TAG" -- "$H" "$S" "$L" "$M" ':!*/results/*' \
  || { echo "[prereg] refusing: $H, $S, $L or $M differs from the tag $TAG"; exit 2; }
WANT=$(sed -n 's/^.*jobs sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$H/jobs/items.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] refusing: jobs sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M35", 11, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M35's ledger plus this pod's upper bound \$$BOUND would pass the \$11 cap"; exit 2; }
mkdir -p "$RES"
LOG="$RES.$M35_POD.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M35 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
