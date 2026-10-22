#!/usr/bin/env bash
# PA-A1: one guarded RunPod pod (A100 80GB) per model and phase. Mirrors persona_affect/a8_rejection/launch.sh and
# persona_affect/common/launch_gen.sh.
#   M=q27 PHASE=gen bash persona_affect/a1_frustration/launch.sh        # pod 1, before the freeze: histories, summaries
#   M=q27 PHASE=capture bash persona_affect/a1_frustration/launch.sh    # pod 2, after the freeze (tag pa-a1-prereg)
#   # no-stock retries: LOG=<path> TRIES=12 WAIT=240 bash persona/operative/retry.sh bash persona_affect/a1_frustration/launch.sh
# env, required: M (q27 | gemma), PHASE (gen | capture)
#      optional: MAXMIN (gen 55 / 70, capture 45 / 50 for q27 / gemma)   MAXCOST (MAXMIN x $1.80/h)
#                GPUS pipe-separated ("NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe")   CLOUD (SECURE)
#                CONTAINER_GB (q27 120, gemma 150)   LIMIT (gen: the first N requests of each split)
#                ITEMS (capture: persona_affect/a1_frustration/jobs/$M_items.jsonl; must sit under jobs/, which ships)
#                OUT (gen: persona_affect/a1_frustration/results/$M; capture: .../results/$M/acts), repo-relative
#                A1_TAG (pa-a1-prereg)   A1_LAUNCHER (uv run python persona_affect/common/launch_wrap.py, which runs
#                infra/launch_run.py; tests point it at a stub)   NAME (pa-a1-$M-$PHASE)
# Order: (1) M and PHASE are checked; (2) for capture, the tag must exist, DESIGN.md, analyze.py and metrics.py must be
# in it and unchanged since, and ITEMS must exist under jobs/: PREREG is then the tag's commit; (3) only then the budget
# check (check_cap A1 $8 against the ledger plus this pod's upper bound), HF_TOKEN from .env and the launch. Any refusal
# exits 2 before the launcher runs. When the launcher returns, common/book_pod.sh books the pod to the ledger (mini A1).
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
M=${M:-} PHASE=${PHASE:-}
D=persona_affect/a1_frustration
case "$M" in
  q27)   DEF_GB=120 GEN_MIN=55 CAP_MIN=45 ;;
  gemma) DEF_GB=150 GEN_MIN=70 CAP_MIN=50 ;;
  *) echo "M must be q27 or gemma, got '$M'" >&2; exit 2 ;;
esac
case "$PHASE" in
  gen)     DEF_MIN=$GEN_MIN DEF_OUT=$D/results/$M ;;
  capture) DEF_MIN=$CAP_MIN DEF_OUT=$D/results/$M/acts ;;
  *) echo "PHASE must be gen or capture, got '$PHASE'" >&2; exit 2 ;;
esac
MAXMIN=${MAXMIN:-$DEF_MIN}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'" >&2; exit 2 ;; esac
MAXMIN=$((10#$MAXMIN))
[ "$MAXMIN" -ge 1 ] || { echo "MAXMIN must be at least 1 minute" >&2; exit 2; }
if [ -n "${LIMIT:-}" ]; then
  case "$LIMIT" in *[!0-9]*) echo "LIMIT must be a whole number of requests, got '$LIMIT'" >&2; exit 2 ;; esac
  [ "$((10#$LIMIT))" -ge 1 ] || { echo "LIMIT must be at least 1" >&2; exit 2; }
fi
OUT=${OUT:-$DEF_OUT}
OUT=${OUT%/}
case "/$OUT/" in //|*/../*) echo "OUT must be a results dir without '..' (the launcher pulls it back to the same path), got '$OUT'" >&2; exit 2 ;; esac

PODENV=(PHASE="$PHASE" M="$M" OUT="$OUT")
if [ "$PHASE" = gen ]; then
  SHIP=(--ship persona_affect --ship welfare_training --ship vigilante/pod --ship infra)
  if [ -n "${LIMIT:-}" ]; then PODENV+=(LIMIT="$((10#$LIMIT))"); fi
else
  TAG=${A1_TAG:-pa-a1-prereg}
  PREREG=$(git rev-parse -q --verify "$TAG^{commit}" 2>/dev/null) \
    || { echo "tag $TAG does not exist: capture runs only after the freeze" >&2; exit 2; }
  for f in DESIGN.md analyze.py metrics.py; do
    git cat-file -e "$PREREG:$D/$f" 2>/dev/null || { echo "$D/$f is not in tag $TAG: the freeze is incomplete" >&2; exit 2; }
  done
  git diff --quiet "$PREREG" -- "$D/DESIGN.md" "$D/analyze.py" "$D/metrics.py" \
    || { echo "DESIGN.md, analyze.py or metrics.py changed since tag $TAG: capture refuses to run on a changed design" >&2; exit 2; }
  ITEMS=${ITEMS:-$D/jobs/${M}_items.jsonl}
  case "$ITEMS" in $D/jobs/*) ;; *) echo "ITEMS must sit under $D/jobs/ (the pod gets what ships; results/ does not), got '$ITEMS'" >&2; exit 2 ;; esac
  [ -f "$ITEMS" ] || { echo "ITEMS not found: $ITEMS (build it with gen items)" >&2; exit 2; }
  SHIP=(--ship persona_affect --ship welfare_training --ship infra)
  PODENV+=(ITEMS="$ITEMS" PREREG="$PREREG")
fi

MAXCOST=${MAXCOST:-$(python3 -c "print(f'{$MAXMIN * 1.80 / 60:.2f}')")}
BOUND=$(bash persona_affect/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST") || { echo "MAXCOST must be a number of USD, got '$MAXCOST'" >&2; exit 2; }
uv run python -c 'import sys; from persona_affect.common.util import check_cap; check_cap("A1", 8.0, planned=float(sys.argv[1]))' "$BOUND" \
  || { echo "[budget] refusing to launch: A1's ledger plus this pod's upper bound \$$BOUND would pass the \$8 cap" >&2; exit 2; }
if [ -f .env ]; then
  ENV_TOKEN=$(sed -n -E 's/^[[:space:]]*(export[[:space:]]+)?HF_TOKEN[[:space:]]*=[[:space:]]*["'\'']?([^"'\''[:space:]]+).*$/\2/p' .env | head -n 1 || true)
  if [ -n "$ENV_TOKEN" ]; then HF_TOKEN=$ENV_TOKEN; fi
fi
export HF_TOKEN=${HF_TOKEN:-}
IFS='|' read -ra GPUARR <<< "${GPUS:-NVIDIA A100-SXM4-80GB|NVIDIA A100 80GB PCIe}"
NAME=${NAME:-pa-a1-$M-$PHASE}
read -ra LAUNCHER <<< "${A1_LAUNCHER:-uv run python persona_affect/common/launch_wrap.py}"

mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
set +e   # whatever the launcher returns, what it printed is still booked
# A signal must not skip the booking: the launcher terminates the pod on INT/TERM/HUP, tee keeps logging through them
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${LAUNCHER[@]}" --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST" \
  --gpus "${GPUARR[@]}" --container-gb "${CONTAINER_GB:-$DEF_GB}" --cloud "${CLOUD:-SECURE}" "${SHIP[@]}" \
  --results-dir "$OUT" -- env "${PODENV[@]}" bash persona_affect/a1_frustration/pod.sh \
  2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?   # pipefail: the launcher's code ($PIPESTATUS is reset by the trap handler on bash 3.2, $? is not)
bash persona_affect/common/book_pod.sh "$LOG" A1 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
