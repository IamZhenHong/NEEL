#!/usr/bin/env bash
# SC-S1 launcher (DESIGN.md §4): one guarded pod per call, booked to self_concept/ledger.jsonl.
#   S1_POD=q bash self_concept/s1_scale/launch.sh     # Qwen3.6-27B then Qwen3.5-9B, one A100 80GB, MAXMIN 330
#   S1_POD=l bash self_concept/s1_scale/launch.sh     # Llama-3.1-70B-Instruct, two A100 80GB, MAXMIN 270
#   SC_S1_LAUNCH_DRY=1 ...                            # print the command only
# The two upper bounds ($9.90 + $16.20) sum to $26.10, inside the $30 cap, so the pods may run at once. Uploads code
# and rows only (the base models download on the pod; Llama is gated, so HF_TOKEN is read from .env). Refuses unless
# the tag sc-s1-prereg exists, the frozen jobs match DESIGN.md's sha256, and check_cap("SC-S1", 30, planned) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/s1_scale; V=persona_affect/v1_value_hook
DRY=${SC_S1_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[s1-launch] refusing to launch a pod under pytest"; exit 2
fi
case "${S1_POD:-}" in
  q) MODELS="Q27 Q9"; NGPU=1; RATE=180; MAXMIN=${MAXMIN:-330}; DISK=220 ;;
  l) MODELS="L70"; NGPU=2; RATE=360; MAXMIN=${MAXMIN:-270}; DISK=260 ;;
  *) echo "set S1_POD=q (Qwen 27B + 9B) or S1_POD=l (Llama 70B)"; exit 2 ;;
esac
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
NAME=sc-s1-$S1_POD; RES=$H/results/main
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$H")
EXTRA=(--exclude "$H/results" --exclude 'data/adapters*')
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpu-count "$NGPU" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud COMMUNITY
     --container-gb "$DISK" "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" S1_MODELS="$MODELS" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[s1-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
[ -n "$HF_TOKEN" ] || { echo "[s1-launch] no HF_TOKEN in .env (Llama is gated)"; exit 2; }
git rev-parse -q --verify refs/tags/sc-s1-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-s1-prereg: commit and tag DESIGN.md and the code first"; exit 2; }
WANT=$(sed -n 's/^.*jobs sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$H/jobs/items.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] jobs sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-S1", 30, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-S1's ledger plus this pod's upper bound \$$BOUND would pass the \$30 cap"; exit 2; }
mkdir -p "$RES"
LOG="$RES.$S1_POD.launch.log"
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-S1 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
