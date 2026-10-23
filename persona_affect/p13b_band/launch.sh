#!/usr/bin/env bash
# PA-P13b: one guarded RunPod pod for Gemma 4 31B (pod.sh), P13's launch.sh pattern (pod B, ruling 3).
#   MAXMIN=220 bash persona_affect/p13b_band/launch.sh
#   no-stock retries: LOG=persona_affect/p13b_band/results/main.launch TRIES=12 WAIT=240 bash persona/operative/retry.sh \
#     env MAXMIN=220 bash persona_affect/p13b_band/launch.sh
# env: MAXMIN (whole minutes, default 220); CLOUD (SECURE: P13's choice).
# NAME pa-p13b, OUT persona_affect/p13b_band/results/main; the launcher pulls OUT back and its output is tee'd to
# $OUT.launch.log. MAXCOST = MAXMIN x $1.80/h, rounded up to the cent. check_cap("P13b", 8.0, planned=MAXCOST) runs
# first. The job files, frozen/means_gemma.npz and DESIGN.md must be committed under the tag pa-p13b-prereg and
# unchanged since. book_pod.sh books the pod (mini P13b, kind gpu) when the launcher returns.
set -euo pipefail
cd "$(dirname "$0")/../.."
export RUNPOD_ANY_CUDA=1
MINI=P13b
D=persona_affect/p13b_band
MAXMIN=${MAXMIN:-220}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2 ;; esac
MAXMIN=$((10#$MAXMIN))
git rev-parse -q --verify refs/tags/pa-p13b-prereg > /dev/null 2>&1 \
  || { echo "[prereg] no tag pa-p13b-prereg: commit and tag DESIGN.md, frozen/ and jobs/ first"; exit 2; }
for f in "$D/DESIGN.md" "$D/frozen/means_gemma.npz" "$D/jobs/main_gemma.jsonl" "$D/jobs/tf_gemma.jsonl" \
         persona_affect/p13_deference/frozen/gemma.json persona_affect/p13_deference/frozen/gemma_dirs.npz; do
  [ -f "$f" ] || { echo "missing $f"; exit 2; }
  case "$f" in "$D"/*)
    git cat-file -e "pa-p13b-prereg:$f" 2> /dev/null || { echo "[prereg] $f is not in the tag pa-p13b-prereg"; exit 2; }
    git diff --quiet pa-p13b-prereg -- "$f" || { echo "[prereg] $f has changed since the tag pa-p13b-prereg"; exit 2; } ;;
  esac
done
CENTS=$(( (MAXMIN * 180 + 59) / 60 ))   # $1.80/h, rounded up to the cent
MAXCOST=$(printf '%d.%02d' $((CENTS / 100)) $((CENTS % 100)))
NAME=pa-p13b
OUT=$D/results/main
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
[ -n "$HF_TOKEN" ] || { echo "google/gemma-4-31B-it is gated on Hugging Face: put HF_TOKEN in .env"; exit 2; }
uv run python -c "from persona_affect.common.util import check_cap; check_cap('$MINI', 8.0, $MAXCOST)" \
  || { echo "[budget] refusing to launch: the P13b ledger plus MAXCOST \$$MAXCOST would pass the \$8 cap"; exit 2; }
mkdir -p "$(dirname "$OUT")"
LOG="$OUT.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" \
  --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 150 \
  --cloud "${CLOUD:-SECURE}" --ship persona_affect --ship vigilante/src --ship vigilante/cases --ship vigilante/pod \
  --ship vigilante/__init__.py --ship infra --results-dir "$OUT" \
  -- env MAXMIN="$MAXMIN" T0="$(date +%s)" bash "$D/pod.sh" \
  2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?   # pipefail: the launcher's code ($PIPESTATUS is reset by the trap handler on bash 3.2, $? is not)
bash persona_affect/common/book_pod.sh "$LOG" "$MINI" "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
