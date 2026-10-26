#!/usr/bin/env bash
# EL-S2 + EL-C3 launcher for deviation 3 (S2 DESIGN.md §11): launch_dev2.sh plus the `f2` mode (the § organism's f2
# only, resumed from results/f2), kept in its own file so no launcher used earlier is edited. One guarded A100 80GB pod (pod.sh), booked to
# evalaware_lens/ledger.jsonl split 55 / 45 (book.sh).
#   EL_POD=smoke bash evalaware_lens/s2_evalself/launch.sh    # the parse / timing check (neutral cue only), MAXMIN 75
#   EL_POD=main  bash evalaware_lens/s2_evalself/launch.sh    # the frozen run, MAXMIN 270
#   EL_POD=sig   bash evalaware_lens/s2_evalself/launch_dev2.sh   # deviation 2: finish the § organism's cell, MAXMIN 110
#   EL_POD=f2    bash evalaware_lens/s2_evalself/launch_dev3.sh   # deviation 3: the § organism's f2 only, MAXMIN 75
#   EL_LAUNCH_DRY=1 ...                                       # print the command only
# GPUs A100-SXM4-80GB or A100 80GB PCIe, 220 GB container; MAXCOST = ceil(MAXMIN x $1.80 / 60) to the cent. V1's
# VN_s1 / NW_s1 / NZ_s1 adapters are hard-linked into data/ad_in and go as --sync-extra files; every jobs/, results/
# and data/adapters* folder stays off the code sync. HF_TOKEN (from .env) reaches the pod as an environment variable.
# Refuses: main without the tags el-s2-prereg and el-c3-prereg, or when the frozen items' jobs_sha differs from S2's
# DESIGN.md; either pod when check_both (each mini's cap on its share of the upper bound, the $226 programme cap, the
# live RunPod balance) fails.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/s2_evalself; V=persona_affect/v1_value_hook; V1C=persona_affect/v1c_selfreport
DRY=${EL_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[s2-launch] refusing to launch a pod under pytest"; exit 2
fi
case "${EL_POD:-}" in
  smoke) DEFMIN=75; SMOKE=1 ;;
  main) DEFMIN=270; SMOKE= ;;
  sig) DEFMIN=110; SMOKE= ;;
  f2) DEFMIN=75; SMOKE= ;;
  *) echo "set EL_POD=smoke or EL_POD=main"; exit 2 ;;
esac
MAXMIN=${MAXMIN:-$DEFMIN}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=el-s2c3-$EL_POD; RES=$H/results/$EL_POD
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
JOBS_SHA=$(sed -n 's/^.*jobs_sha: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship self_concept/s1_scale
       --ship self_concept/n4_conditional --ship evalaware_lens/__init__.py --ship evalaware_lens/common --ship "$H"
       --ship evalaware_lens/c3_masking)
EXTRA=()
SIGONLY=; EXPECT_FORCED_MISSING=12824; EXPECT_F2=
if [ "$EL_POD" = sig ] || [ "$EL_POD" = f2 ]; then   # the § organism's partial files, which evalrun resumes from
  SIGONLY=1
  [ "$EL_POD" = f2 ] && { EXPECT_FORCED_MISSING=0; EXPECT_F2=14080; }
  for f in eval_SIG.jsonl forced_SIG.jsonl; do EXTRA+=(--sync-extra "$RES/$f"); done
else
  for a in VN_s1 NW_s1 NZ_s1; do
    for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$H/data/ad_in/$a/$f"); done
  done
fi
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results \
         evalaware_lens/*/results "$H/data"; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
EXTRA+=(--exclude 'data/adapters*' --exclude "$H/results/main/ad_*")
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
CMD=(uv run python persona_affect/common/launch_wrap.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud SECURE --container-gb 220
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" JOBS_SHA="$JOBS_SHA" SMOKE="$SMOKE" SIGONLY="$SIGONLY"
     EXPECT_FORCED_MISSING="$EXPECT_FORCED_MISSING" EXPECT_F2="$EXPECT_F2" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[s2-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
[ -n "$JOBS_SHA" ] || { echo "[prereg] no jobs_sha in $H/DESIGN.md"; exit 2; }
HAVE=$(uv run python -c "from evalaware_lens.s2_evalself import items as I; print(I.jobs_sha(I.build()))")
[ "$HAVE" = "$JOBS_SHA" ] || { echo "[prereg] jobs_sha $HAVE does not match DESIGN.md ($JOBS_SHA)"; exit 2; }
if [ "$EL_POD" = f2 ]; then
  git rev-parse -q --verify refs/tags/el-s2-dev3 > /dev/null 2>&1 || { echo "[prereg] no tag el-s2-dev3"; exit 2; }
  for f in eval_SIG.jsonl forced_SIG.jsonl; do [ -s "$RES/$f" ] || { echo "[s2-launch] missing $RES/$f"; exit 2; }; done
fi
if [ "$EL_POD" = sig ]; then
  git rev-parse -q --verify refs/tags/el-s2-dev2 > /dev/null 2>&1 || { echo "[prereg] no tag el-s2-dev2"; exit 2; }
  for f in eval_SIG.jsonl forced_SIG.jsonl; do [ -s "$RES/$f" ] || { echo "[s2-launch] missing $RES/$f"; exit 2; }; done
fi
if [ "$EL_POD" = main ]; then
  for t in el-s2-prereg el-c3-prereg; do
    git rev-parse -q --verify "refs/tags/$t" > /dev/null 2>&1 || { echo "[prereg] no tag $t"; exit 2; }
  done
fi
for a in VN_s1 NW_s1 NZ_s1; do
  src=$V/data/adapters/$a; [ "$a" = VN_s1 ] || src=$V1C/data/adapters/$a
  mkdir -p "$H/data/ad_in/$a"
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$src/$f" ] || { echo "[s2-launch] missing $src/$f"; exit 2; }
    ln -f "$src/$f" "$H/data/ad_in/$a/$f"
  done
done
BOUND=$(bash "$H/book.sh" --bound "$MAXMIN" "$MAXCOST")
CAPCHECK='import sys; from evalaware_lens.common import ledger as LG; from evalaware_lens.s2_evalself.book import SHARE
b = float(sys.argv[1])
LG.check_both("EL-S2", 8.0, planned=round(b * SHARE["EL-S2"], 2), kind="gpu")
LG.check_both("EL-C3", 6.0, planned=round(b * SHARE["EL-C3"], 2), kind="gpu")
print(f"[budget] ok: bound ${b:.2f} = S2 ${b * SHARE["EL-S2"]:.2f} + C3 ${b * SHARE["EL-C3"]:.2f}")'
uv run python -c "$CAPCHECK" "$BOUND" || { echo "[budget] refusing to launch (bound \$$BOUND)"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the launcher terminates the pod, then this books what its log shows"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash "$H/book.sh" "$LOG" EL-S2+EL-C3 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
