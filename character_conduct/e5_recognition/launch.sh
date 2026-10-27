#!/usr/bin/env bash
# CC-E5 step 1 launcher: one guarded A100 80GB pod (pod.sh), run by infra/launch_run.py directly (detached-safe: a
# laptop TERM leaves the pod and its job running for a manual pull), booked to character_conduct/ledger.jsonl as CC-E5.
#   bash character_conduct/e5_recognition/launch.sh                  # MAXMIN 150: max cost $4.50
#   E5_LAUNCH_DRY=1 bash character_conduct/e5_recognition/launch.sh  # print the command only
# Refuses unless the tag cc-e5-prereg exists, e5_recognition's code matches HEAD, the frozen files are unchanged since
# the tag, the jobs rebuilt here match DESIGN.md's jobs_sha, and LG.check_both("CC-E5", CAP, planned = the bound) passes.
# The pod fetches ODCV-Bench itself; data/ (the local clone) and results/ stay off the code sync.
set -euo pipefail
cd "$(dirname "$0")/../.."
E=character_conduct/e5_recognition
DRY=${E5_LAUNCH_DRY:-}
MAXMIN=${MAXMIN:-150}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
CAP=60
export RUNPOD_ANY_CUDA=1
NAME=cc-e5-reads; RES=$E/results/reads
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
JOBS_SHA=$(sed -n 's/^.*jobs_sha: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$E/DESIGN.md" | head -1)
SHIP=(--ship character_conduct/__init__.py --ship "$E" --ship jlens/__init__.py --ship jlens/e2_realism/__init__.py
      --ship jlens/e2_realism/lens_core.py --ship infra)
EXTRA=(--exclude "$E/data" --exclude "$E/results" --exclude __pycache__)
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud SECURE --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env OUT="$RES" JOBS_SHA="$JOBS_SHA" bash "$E/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[e5-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}" | sed 's/HF_TOKEN=[^ ]*/HF_TOKEN=***/'
  exit 0
fi
[ -n "$HF_TOKEN" ] || { echo "[e5-launch] no HF_TOKEN in .env"; exit 2; }
git rev-parse -q --verify refs/tags/cc-e5-prereg > /dev/null 2>&1 || { echo "[prereg] no tag cc-e5-prereg"; exit 2; }
git diff --quiet HEAD -- "$E" ":!$E/results" ":!$E/results/*" ":!$E/data" ":!$E/data/*" \
  || { echo "[prereg] $E differs from HEAD: commit (a deviation note) first"; exit 2; }
git diff --quiet cc-e5-prereg -- "$E/DESIGN.md" "$E/build.py" "$E/traj.py" "$E/tokensets.py" "$E/pod_reads.py" "$E/analyze.py" \
  || { echo "[prereg] a frozen file changed since cc-e5-prereg"; exit 2; }
[ -n "$JOBS_SHA" ] || { echo "[prereg] no jobs_sha in $E/DESIGN.md"; exit 2; }
HAVE=$(HF_HUB_OFFLINE=1 uv run python -c "from character_conduct.e5_recognition import build as B; print(B.jobs_sha(B.build()))" 2>/dev/null | tail -1)
[ "$HAVE" = "$JOBS_SHA" ] || { echo "[prereg] jobs_sha $HAVE does not match DESIGN.md ($JOBS_SHA)"; exit 2; }
BOUND=$(bash "$E/book_pod.sh" --bound "$MAXMIN" "$MAXCOST")
uv run python -c 'import sys; import character_conduct.common.ledger as LG; LG.check_both("CC-E5", float(sys.argv[2]), planned=float(sys.argv[1]), kind="gpu")' "$BOUND" "$CAP" \
  || { echo "[budget] refusing to launch: check_both failed for CC-E5 (cap \$$CAP, bound \$$BOUND)"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the pod keeps running (launch_run, detached-safe); pull by hand, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash "$E/book_pod.sh" "$LOG" CC-E5 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
