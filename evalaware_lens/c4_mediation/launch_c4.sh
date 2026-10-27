#!/usr/bin/env bash
# EL-C4 launcher: one guarded 2 x A100 80GB pod, run by infra/launch_run.py directly (detached-safe: a laptop TERM
# leaves the pod and its job running for the manual pull), booked to evalaware_lens/ledger.jsonl as EL-C4.
#   bash evalaware_lens/c4_mediation/launch_c4.sh            # MAXMIN 250: max cost $14.92
#   EL_C4_LAUNCH_DRY=1 bash evalaware_lens/c4_mediation/launch_c4.sh   # print the command only
# Ships: evalaware_lens/__init__.py, c2_steer (code + data; its results/ are dropped by the launcher's './results'
# exclude), c4_mediation, F2's dirs.py and job files; C2's row-0 replies (results/wood/gap.jsonl) by --sync-extra.
# Refuses unless the tag el-c4-prereg exists, c4_mediation and c2_steer's code match HEAD, C2's items match DESIGN.md's
# sha256, and check_both("EL-C4", 21, planned = the bound, kind="gpu") passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/c4_mediation; C2=evalaware_lens/c2_steer
DRY=${EL_C4_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[c4-launch] refusing to launch a pod under pytest"; exit 2
fi
MAXMIN=${MAXMIN:-250}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
GPUS=("NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe"); NGPU=2; RATE=358
export RUNPOD_ANY_CUDA=1
NAME=el-c4; RES=$H/results/main
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
SHIP=(--ship evalaware_lens/__init__.py --ship "$C2" --ship "$H" --ship evalaware_lens/f2_panel/__init__.py
      --ship evalaware_lens/f2_panel/dirs.py --ship evalaware_lens/f2_panel/jobs)
EXTRA=(--exclude "$H/results" --exclude "$C2/results" --exclude 'data/adapters*' --exclude __pycache__
       --sync-extra "$C2/results/wood/gap.jsonl")
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpu-count "$NGPU" --gpus "${GPUS[@]}" --cloud COMMUNITY --container-gb 220 "${SHIP[@]}" "${EXTRA[@]}"
     --results-dir "$RES" -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" bash "$H/pod_c4.sh")
if [ "$DRY" = 1 ]; then
  echo "[c4-launch] dry run (nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
[ -n "$HF_TOKEN" ] || { echo "[c4-launch] no HF_TOKEN in .env"; exit 2; }
git rev-parse -q --verify refs/tags/el-c4-prereg > /dev/null 2>&1 || { echo "[prereg] no tag el-c4-prereg"; exit 2; }
git diff --quiet HEAD -- "$H" "$C2" ":!$H/results" ":!$H/results/*" ":!$C2/results" ":!$C2/results/*" \
  || { echo "[prereg] $H or $C2 differs from HEAD: commit (a deviation note) first"; exit 2; }
git diff --quiet el-c4-prereg -- "$H/DESIGN.md" "$H/rules.py" "$H/analyze.py" \
  || { echo "[prereg] DESIGN.md, rules.py or analyze.py changed since el-c4-prereg"; exit 2; }
WANT=$(sed -n 's/^.*items sha256 `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
HAVE=$(shasum -a 256 "$C2/data/items.jsonl.gz" | cut -d' ' -f1)
[ -n "$WANT" ] && [ "$WANT" = "$HAVE" ] || { echo "[prereg] C2 items sha256 $HAVE does not match DESIGN.md ($WANT)"; exit 2; }
[ -f "$C2/results/wood/gap.jsonl" ] || { echo "[c4-launch] $C2/results/wood/gap.jsonl missing"; exit 2; }
BOUND=$(bash evalaware_lens/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c 'import sys; from evalaware_lens.common import ledger as LG; LG.check_both("EL-C4", 21.0, planned=float(sys.argv[1]), kind="gpu")' "$BOUND" \
  || { echo "[budget] refusing to launch: check_both failed for EL-C4 (cap \$21, bound \$$BOUND)"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the pod keeps running (launch_run, detached-safe); pull by hand, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash evalaware_lens/common/book_pod.sh "$LOG" EL-C4 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
