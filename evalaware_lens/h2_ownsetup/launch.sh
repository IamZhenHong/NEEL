#!/usr/bin/env bash
# EL-H2 launcher: one guarded A100 80GB pod (pod.sh), run by infra/launch_run.py directly (detached-safe: a laptop TERM
# leaves the pod and its job running for the manual pull), booked to evalaware_lens/ledger.jsonl as EL-H2.
#   bash evalaware_lens/h2_ownsetup/launch.sh                 # MAXMIN 100: max cost $3.00
#   EL_H2_LAUNCH_DRY=1 bash evalaware_lens/h2_ownsetup/launch.sh    # print the command only
# V1's VN_s1 and EL-S2's NACT_s1 / NACTZ_s1 are hard-linked into data/ad_in and go as --sync-extra files; every jobs/,
# results/ and data/adapters* folder stays off the code sync. HF_TOKEN (from .env) reaches the pod as an environment
# variable. Refuses unless the tag el-h2-prereg exists, h2_ownsetup's code matches HEAD, the items' jobs_sha matches
# DESIGN.md, the adapters' sha256 match EL-S2's manifest, and check_both("EL-H2", 8, planned = the bound) passes.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=evalaware_lens/h2_ownsetup; S2=evalaware_lens/s2_evalself; V=persona_affect/v1_value_hook
DRY=${EL_H2_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then
  echo "[h2-launch] refusing to launch a pod under pytest"; exit 2
fi
MAXMIN=${MAXMIN:-100}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes, got '$MAXMIN'"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=el-h2; RES=$H/results/main
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
JOBS_SHA=$(sed -n 's/^.*jobs_sha: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship self_concept/s1_scale
       --ship self_concept/n4_conditional --ship evalaware_lens/__init__.py --ship evalaware_lens/common
       --ship "$S2" --ship evalaware_lens/c3_masking --ship "$H")
EXTRA=()
for a in VN_s1 NACT_s1 NACTZ_s1; do
  for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$H/data/ad_in/$a/$f"); done
done
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results \
         evalaware_lens/*/results "$H/data" "$S2/data"; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
EXTRA+=(--exclude 'data/adapters*' --exclude __pycache__)
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud SECURE --container-gb 150
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" JOBS_SHA="$JOBS_SHA" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then
  echo "[h2-launch] dry run (nothing staged, nothing checked, no pod):"
  echo "${CMD[*]}"
  exit 0
fi
[ -n "$HF_TOKEN" ] || { echo "[h2-launch] no HF_TOKEN in .env"; exit 2; }
git rev-parse -q --verify refs/tags/el-h2-prereg > /dev/null 2>&1 || { echo "[prereg] no tag el-h2-prereg"; exit 2; }
git diff --quiet HEAD -- "$H" ":!$H/results" ":!$H/results/*" ":!$H/data" ":!$H/data/*" \
  || { echo "[prereg] $H differs from HEAD: commit (a deviation note) first"; exit 2; }
git diff --quiet el-h2-prereg -- "$H/DESIGN.md" "$H/items.py" "$H/analyze.py" \
  || { echo "[prereg] DESIGN.md, items.py or analyze.py changed since el-h2-prereg"; exit 2; }
[ -n "$JOBS_SHA" ] || { echo "[prereg] no jobs_sha in $H/DESIGN.md"; exit 2; }
HAVE=$(uv run python -c "from evalaware_lens.h2_ownsetup import items as I; print(I.jobs_sha(I.build()))")
[ "$HAVE" = "$JOBS_SHA" ] || { echo "[prereg] jobs_sha $HAVE does not match DESIGN.md ($JOBS_SHA)"; exit 2; }
for a in VN_s1 NACT_s1 NACTZ_s1; do
  src=$S2/results/main/ad_VN/$a; [ "$a" = VN_s1 ] && src=$V/data/adapters/$a
  mkdir -p "$H/data/ad_in/$a"
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$src/$f" ] || { echo "[h2-launch] missing $src/$f"; exit 2; }
    ln -f "$src/$f" "$H/data/ad_in/$a/$f"
  done
done
uv run python - <<'EOF' || { echo "[prereg] adapter sha256 mismatch"; exit 2; }
import hashlib, json, re
from pathlib import Path
h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
d = Path("evalaware_lens/h2_ownsetup/DESIGN.md").read_text()
for a in ("VN_s1", "NACT_s1", "NACTZ_s1"):
    want = re.search(rf"`{a}` sha256 `([0-9a-f]{{64}})`", d).group(1)
    have = h(f"evalaware_lens/h2_ownsetup/data/ad_in/{a}/adapter_model.safetensors")
    assert have == want, (a, have, want)
print("[prereg] adapter sha256 ok")
EOF
BOUND=$(bash evalaware_lens/common/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c 'import sys; from evalaware_lens.common import ledger as LG; LG.check_both("EL-H2", 8.0, planned=float(sys.argv[1]), kind="gpu")' "$BOUND" \
  || { echo "[budget] refusing to launch: check_both failed for EL-H2 (cap \$8, bound \$$BOUND)"; exit 2; }
mkdir -p "$RES"
LOG=$RES.launch.log
set +e
trap 'echo "[launch] interrupted: the pod keeps running (launch_run, detached-safe); pull by hand, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash evalaware_lens/common/book_pod.sh "$LOG" EL-H2 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
