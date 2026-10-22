#!/usr/bin/env bash
# PA-P13: one guarded RunPod pod, A (calibration: pod_a.sh) or B (main: pod_b.sh), for one model. It copies
# persona_affect/common/launch_gen.sh's pattern (ruling 3); only the pod command, the names and the input checks differ.
#   WHICH=a M=gemma MAXMIN=55 bash persona_affect/p13_deference/launch.sh
#   WHICH=b M=gemma MAXMIN=130 bash persona_affect/p13_deference/launch.sh
#   WHICH=b M=q27 MAXMIN=35 TF_ONLY=1 bash persona_affect/p13_deference/launch.sh       # q27 is teacher-forced only
#   # no-stock retries
#   LOG=persona_affect/p13_deference/launch_a_gemma TRIES=12 WAIT=240 bash persona/operative/retry.sh \
#     env WHICH=a M=gemma MAXMIN=55 bash persona_affect/p13_deference/launch.sh
# env: WHICH (a | b), M (gemma | q27), MAXMIN (whole minutes); TF_ONLY (pod B; required for q27); CLOUD (SECURE).
# NAME is pa-p13-$WHICH-$M and OUT persona_affect/p13_deference/results/${WHICH}_$M; the launcher pulls OUT back and
# its output is tee'd to $OUT.launch.log. MAXCOST = MAXMIN x $1.80/h, rounded up to the cent; the guard refuses a pod
# whose rate x MAXMIN passes it. check_cap("P13", 10.0, planned=MAXCOST) runs before the launch.
# Before launching it checks the pod's inputs here: pod A's job files (and pod_gen --check on the calibration jobs),
# pod B's job files and frozen files, which must be committed under the tag pa-p13-prereg and unchanged since (the
# prereg comes before any outcome data, Task 6 Step 5). The pod gets T0 (now), from which its deadline is counted.
# When the launcher returns, persona_affect/common/book_pod.sh books the pod (mini P13, kind gpu): the cost on the
# launcher's "[runpod] terminated <id>" line, an upper bound if a pod was created but never confirmed terminated, and
# nothing if no pod was created (no stock, a refusal), so retry.sh rounds cost nothing.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
MINI=P13
D=persona_affect/p13_deference
: "${WHICH:?set WHICH (a or b)}" "${M:?set M (gemma or q27)}" "${MAXMIN:?set MAXMIN (whole minutes)}"
case "$WHICH" in a|b) ;; *) echo "WHICH must be a or b, got '$WHICH'"; exit 2 ;; esac
case "$M" in
  gemma) MODEL=google/gemma-4-31B-it FMT=gemma4 CONTAINER_GB=150 ;;
  q27)   MODEL=Qwen/Qwen3.6-27B FMT=qwen CONTAINER_GB=120 ;;
  *) echo "M must be gemma or q27, got '$M'"; exit 2 ;;
esac
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2 ;; esac
MAXMIN=$((10#$MAXMIN))
[ "$MAXMIN" -gt 0 ] || { echo "MAXMIN must be at least 1 minute, got $MAXMIN"; exit 2; }
if [ "$WHICH" = b ] && [ "$M" = q27 ] && [ -z "${TF_ONLY:-}" ]; then
  echo "q27 is teacher-forced only (accepted change 6): set TF_ONLY=1 for its pod B"; exit 2
fi
if [ "$WHICH" = a ] && [ -n "${TF_ONLY:-}" ]; then echo "TF_ONLY applies to pod B only"; exit 2; fi
if [ "$WHICH" = a ]; then
  NEED="$D/jobs/cal_$M.jsonl $D/jobs/alpha_$M.jsonl"
else
  NEED="$D/jobs/main_$M.jsonl $D/jobs/tf_$M.jsonl $D/jobs/read_$M.jsonl $D/frozen/$M.json $D/frozen/${M}_dirs.npz"
fi
for f in $NEED; do
  [ -f "$f" ] || { echo "missing $f: write it first (jobs cal / calibrate / jobs main)"; exit 2; }
done
if [ "$WHICH" = b ]; then
  git rev-parse -q --verify refs/tags/pa-p13-prereg > /dev/null 2>&1 \
    || { echo "[prereg] no tag pa-p13-prereg: commit and tag DESIGN.md, frozen/ and jobs/ before pod B"; exit 2; }
  for f in "$D/frozen/$M.json" "$D/frozen/${M}_dirs.npz"; do
    git cat-file -e "pa-p13-prereg:$f" 2> /dev/null || { echo "[prereg] $f is not in the tag pa-p13-prereg"; exit 2; }
    git diff --quiet pa-p13-prereg -- "$f" || { echo "[prereg] $f has changed since the tag pa-p13-prereg"; exit 2; }
  done
fi
CENTS=$(( (MAXMIN * 180 + 59) / 60 ))   # $1.80/h, rounded up to the cent
MAXCOST=$(printf '%d.%02d' $((CENTS / 100)) $((CENTS % 100)))
NAME=pa-p13-$WHICH-$M
OUT=$D/results/${WHICH}_$M
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
case "$MODEL" in google/gemma-*)
  [ -n "$HF_TOKEN" ] || { echo "MODEL $MODEL is gated on Hugging Face: put HF_TOKEN in .env"; exit 2; } ;;
esac
if [ "$WHICH" = a ]; then
  uv run python -m persona_affect.common.pod_gen --register persona_affect.p13_deference.cases \
    --check "$D/jobs/cal_$M.jsonl" --fmt "$FMT" \
    || { echo "[pre] refusing to launch: pod_gen --check found jobs that would not run on this pod"; exit 2; }
fi
uv run python -c "from persona_affect.common.util import check_cap; check_cap('P13', 10.0, $MAXCOST)" \
  || { echo "[budget] refusing to launch: the P13 ledger plus MAXCOST \$$MAXCOST would pass the \$10 cap"; exit 2; }
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
# As in launch_gen.sh: a signal must not skip the booking. The launcher (through launch_wrap.py) takes INT, TERM and HUP
# as Ctrl-C and terminates the pod; tee ignores all three, so the log still gets its terminated line.
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" \
  --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb "$CONTAINER_GB" \
  --cloud "${CLOUD:-SECURE}" --ship persona_affect --ship vigilante/src --ship vigilante/cases --ship vigilante/pod \
  --ship vigilante/__init__.py --ship infra --results-dir "$OUT" \
  -- env M="$M" MAXMIN="$MAXMIN" TF_ONLY="${TF_ONLY:-}" T0="$(date +%s)" bash "$D/pod_$WHICH.sh" \
  2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?   # pipefail: the launcher's code ($PIPESTATUS is reset by the trap handler on bash 3.2, $? is not)
bash persona_affect/common/book_pod.sh "$LOG" "$MINI" "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
