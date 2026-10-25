#!/usr/bin/env bash
# SC-M31 launcher (DESIGN.md §4): one guarded pod (one A100 80GB, MAXMIN 165, bound $5.85), booked to
# self_concept/ledger.jsonl.
#   bash self_concept/m31_aianchor/launch.sh
#   SC_M31_LAUNCH_DRY=1 bash self_concept/m31_aianchor/launch.sh       # print the command only
# Uploads code and rows only; the gated base downloads on the pod with HF_TOKEN read from .env (passed in the pod's
# environment, never printed). Refuses unless the tag sc-m31-rerun1 exists, the code matches it, the frozen jobs match
# DESIGN.md's sha256, and check_cap("SC-M31", 10.50, planned) passes. Rerun 1 (DESIGN.md, deviations): run 1's results
# were lost, so the owner raised the cap from $6 to about $10 (run 1's $4.63 + the bound $5.85 = $10.48 -> $10.50); the
# tag moved from sc-m31-prereg to sc-m31-rerun1, which differs from it only in this file and DESIGN.md.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=self_concept/m31_aianchor; S=self_concept/s1_scale; M=self_concept/m1_mediation; L=self_concept/m9_llama8b; V=persona_affect/v1_value_hook
P=self_concept/m20_persona
DRY=${SC_M31_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[m31-launch] refusing to launch a pod under pytest"; exit 2
fi
MAXMIN=${MAXMIN:-165}; RATE=180
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
NAME=sc-m31; RES=$H/results/main
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship "$S" --ship "$M" --ship "$L" --ship "$P" --ship "$H")
EXTRA=(--exclude "$H/results" --exclude "$S/results" --exclude "$M/results" --exclude "$L/results" --exclude "$P/results" --exclude 'data/adapters*')
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" --cloud COMMUNITY
     --container-gb 120 "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[m31-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
[ -n "$HF_TOKEN" ] || { echo "[m31-launch] no HF_TOKEN in .env (Llama is gated)"; exit 2; }
git rev-parse -q --verify refs/tags/sc-m31-rerun1 > /dev/null 2>&1 \
  || { echo "[prereg] no tag sc-m31-rerun1: commit and tag DESIGN.md and the code first"; exit 2; }
git diff --quiet sc-m31-rerun1 -- "$H" "$S" "$M" "$L" "$P" ':!*/results/*' \
  || { echo "[prereg] $H, $S, $M, $L or $P differs from the tag sc-m31-rerun1"; exit 2; }
WANT=$(sed -n 's/^.*jobs sha256: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$S/jobs/items.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] jobs sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
BOUND=$(bash self_concept/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from self_concept.common.ledger import check_cap; check_cap("SC-M31", 10.50, planned=float(sys.argv[1]))'
uv run python -c "$CAPCHECK" "$BOUND" \
  || { echo "[budget] refusing to launch: SC-M31's ledger plus this pod's upper bound \$$BOUND would pass the \$10.50 cap"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash self_concept/common/book_pod.sh "$LOG" SC-M31 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
