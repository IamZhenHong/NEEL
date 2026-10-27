#!/usr/bin/env bash
# CC-E2 stage 2b launcher (DESIGN2b.md §4, §6): one guarded A100 80GB pod (pod.sh), booked to character_conduct/ledger.jsonl.
#   CC_POD=main bash character_conduct/e2_selfuse/launch2b.sh    # the frozen run, MAXMIN 90
#   CC_LAUNCH_DRY=1 ...                                          # print the command only
# infra/launch_run.py is called directly (not launch_wrap), so a laptop memory-cleanup TERM leaves the pod and its
# setsid job running for the manual ssh pull (EL-W2 / H1 lesson). The six habit adapters are hard-linked into data/ad_in and go as --sync-extra files; jobs/, results/ and
# data/adapters* folders stay off the code sync. HF_TOKEN (from .env) reaches the pod as an environment variable, never
# a file or a log line. Refuses: without the tag cc-e2s1-prereg, when the frozen items' jobs_sha differs from DESIGN.md,
# or when check_both (CC-E2's runaway cap, the live RunPod balance) fails.
set -euo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e2_selfuse; V=persona_affect/v1_value_hook
DRY=${CC_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then echo "[e2-launch] refusing to launch under pytest"; exit 2; fi

MAXMIN=${MAXMIN:-90}
case "$MAXMIN" in ''|*[!0-9]*) echo "MAXMIN must be a whole number of minutes"; exit 2;; esac
export RUNPOD_ANY_CUDA=1
NAME=cc-e2s2b-main; RES=$H/results/stage2b
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * 180 / 60) / 100:.2f}')" "$MAXMIN")
JOBS_SHA=$(sed -n 's/^.*jobs_sha: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN2b.md" | head -1)
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship self_concept/s1_scale
       --ship self_concept/n4_conditional --ship evalaware_lens/__init__.py --ship evalaware_lens/common
       --ship evalaware_lens/s2_evalself/__init__.py --ship evalaware_lens/s2_evalself/evalrun.py
       --ship evalaware_lens/s2_evalself/items.py --ship character_conduct/__init__.py --ship character_conduct/common
       --ship "$H")
src_of() {   # the on-disk adapter folder of a shipped arm (bash 3.2: no associative arrays)
  case "$1" in
    VN_s1) echo "$V/data/adapters/VN_s1" ;;
    NW_s1|NZ_s1) echo "$V/results/full_s1/ad_later/$1" ;;
    VN_s2) echo "$V/results/full_s2/ad/VN_s2" ;;
    NW_s2|NZ_s2) echo "$V/results/full_s2/ad_later/$1" ;;
  esac
}
EXTRA=()
for a in VN_s1 NW_s1 NZ_s1 VN_s2 NW_s2 NZ_s2; do
  for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$H/data/ad_in/$a/$f"); done
done
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results \
         evalaware_lens/*/results character_conduct/*/results "$H/data"; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
EXTRA+=(--exclude 'data/adapters*')
HF_TOKEN="$(uv run python -c "import re; m = re.search(r'HF_TOKEN\s*=\s*[\"\']?([^\"\'\s]+)', open('.env').read()); print(m.group(1) if m else '')")"
export HF_TOKEN
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN"
     --max-cost "$MAXCOST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --cloud SECURE --container-gb 220
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" JOBS_SHA="$JOBS_SHA" bash "$H/pod2b.sh")
if [ "$DRY" = 1 ]; then echo "[e2-launch] dry run:"; echo "${CMD[*]}" | sed 's/HF_TOKEN=[^ ]*/HF_TOKEN=***/'; exit 0; fi
[ -n "$JOBS_SHA" ] || { echo "[prereg] no jobs_sha in $H/DESIGN.md"; exit 2; }
HAVE=$(uv run python -c "from character_conduct.e2_selfuse import items2 as I2, items2b as I2B; print(I2.jobs_sha(I2B.read_jobs()))")
[ "$HAVE" = "$JOBS_SHA" ] || { echo "[prereg] jobs_sha $HAVE does not match DESIGN.md ($JOBS_SHA)"; exit 2; }
git rev-parse -q --verify refs/tags/cc-e2s2b-prereg > /dev/null 2>&1 || { echo "[prereg] no tag cc-e2s2b-prereg"; exit 2; }
for a in VN_s1 NW_s1 NZ_s1 VN_s2 NW_s2 NZ_s2; do
  src=$(src_of "$a")
  mkdir -p "$H/data/ad_in/$a"
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$src/$f" ] || { echo "[e2-launch] missing $src/$f"; exit 2; }
    ln -f "$src/$f" "$H/data/ad_in/$a/$f"
  done
done
BOUND=$(bash "$H/book.sh" --bound "$MAXMIN" "$MAXCOST")
uv run python -c "import sys; from character_conduct.common import ledger as LG; LG.check_both('CC-E2', 40.0, planned=float(sys.argv[1]), kind='gpu'); print(f'[budget] ok: bound \${float(sys.argv[1]):.2f}')" "$BOUND" \
  || { echo "[budget] refusing to launch (bound \$$BOUND)"; exit 2; }
mkdir -p "$(dirname "$RES")"
LOG="$RES.launch.log"
set +e
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash "$H/book.sh" "$LOG" CC-E2 "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
