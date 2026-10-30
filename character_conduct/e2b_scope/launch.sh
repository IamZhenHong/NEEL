#!/usr/bin/env bash
# CC-E2b launcher (DESIGN.md §4, §6): one guarded A100 SXM4 80GB pod (pod.sh), booked to character_conduct/ledger.jsonl
# as CC-E2b. MAXMIN = the smaller of 50 and the minutes left before 18:35 HKT; refuses under 35. At 179 c/hr and MAXMIN
# 50: MAXCOST $1.50, bound $2.40 = 80% of the $3 cap. SECURE cloud unless E2B_CLOUD=COMMUNITY; SXM4 only (the PCIe host
# 104.255.9.187 can't be picked).
#   bash character_conduct/e2b_scope/launch.sh          # E2B_LAUNCH_DRY=1 prints the command only
# Ships E2 stage 1's code paths + this folder (results, jobs and adapters excluded) and the three adapters (--sync-extra).
# Refuses unless the tag cc-e2b-prereg exists, the frozen items' jobs_sha equals DESIGN.md's, the adapters match
# DESIGN.md's sha256s, the GPU spend so far + the bound stays within $2.40, and check_both passes (CC-E2b, $3). No HF
# token is sent (Qwen3.5-9B is public).
set -euo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e2b_scope; E2=character_conduct/e2_selfuse; V=persona_affect/v1_value_hook
AD=$E2/data/ad_in
DRY=${E2B_LAUNCH_DRY:-}
if [ -n "${PYTEST_CURRENT_TEST:-}" ] && [ "$DRY" != 1 ]; then echo "[e2b-launch] refusing to launch under pytest"; exit 2; fi
export RUNPOD_ANY_CUDA=1
NAME=cc-e2b; RES=$H/results/pod; RATE=179; GPU_STOP=2.40; CLOUD=${E2B_CLOUD:-SECURE}
LEFT=$(( ( $(TZ=Asia/Hong_Kong date -j -f "%Y-%m-%d %H:%M" "$(TZ=Asia/Hong_Kong date +%Y-%m-%d) 18:35" +%s) - $(date +%s) ) / 60 ))
MAXMIN=$(( LEFT < 50 ? LEFT : 50 ))
[ "$MAXMIN" -ge 35 ] || { echo "[e2b-launch] only $MAXMIN min before 18:35 HKT; not launching"; exit 2; }
MAXCOST=$(python3 -c "import math, sys; print(f'{math.ceil(int(sys.argv[1]) * int(sys.argv[2]) / 60) / 100:.2f}')" "$MAXMIN" "$RATE")
JOBS_SHA=$(sed -n 's/^.*jobs_sha: `\([0-9a-f]\{64\}\)`.*$/\1/p' "$H/DESIGN.md" | head -1)
SHIP=()
while read -r p; do [ -z "$p" ] || SHIP+=(--ship "$p"); done < "$V/ship.txt"
SHIP+=(--ship self_concept/__init__.py --ship self_concept/common --ship self_concept/s1_scale
       --ship self_concept/n4_conditional --ship evalaware_lens/__init__.py --ship evalaware_lens/common
       --ship evalaware_lens/s2_evalself/__init__.py --ship evalaware_lens/s2_evalself/evalrun.py
       --ship evalaware_lens/s2_evalself/items.py --ship character_conduct/__init__.py --ship character_conduct/common
       --ship character_conduct/e1a_adapter/__init__.py --ship character_conduct/e1a_adapter/analyze.py
       --ship "$E2" --ship "$H")
EXTRA=()
for a in VN_s1 NACT_s1 NACTZ_s1; do
  for f in adapter_model.safetensors adapter_config.json; do EXTRA+=(--sync-extra "$AD/$a/$f"); done
done
for d in persona_affect/*/jobs persona_affect/*/data/adapters* persona_affect/*/results self_concept/*/results \
         evalaware_lens/*/results character_conduct/*/results "$E2/data" "$E2/jobs"; do
  [ -d "$d" ] && EXTRA+=(--exclude "$d")
done
EXTRA+=(--exclude 'data/adapters*' --exclude '__pycache__')
CMD=(uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MAXMIN" --max-cost "$MAXCOST"
     --gpus "NVIDIA A100-SXM4-80GB" --cloud "$CLOUD" --container-gb 150 --hf-token ""
     "${SHIP[@]}" "${EXTRA[@]}" --results-dir "$RES"
     -- env T0="$(date +%s)" MAXMIN="$MAXMIN" OUT="$RES" JOBS_SHA="$JOBS_SHA" bash "$H/pod.sh")
if [ "$DRY" = 1 ]; then echo "[e2b-launch] dry run (nothing checked, no pod):"; echo "${CMD[*]}"; exit 0; fi
[ -n "$JOBS_SHA" ] || { echo "[prereg] no jobs_sha in $H/DESIGN.md"; exit 2; }
git rev-parse -q --verify refs/tags/cc-e2b-prereg > /dev/null 2>&1 || { echo "[prereg] no tag cc-e2b-prereg"; exit 2; }
HAVE=$(uv run python -c "from character_conduct.e2b_scope import items as I; print(I.jobs_sha(I.build()))")
[ "$HAVE" = "$JOBS_SHA" ] || { echo "[prereg] jobs_sha $HAVE does not match DESIGN.md ($JOBS_SHA)"; exit 2; }
[ ! -e "$RES/sampled_VN-NACT_s1.jsonl" ] || { echo "[e2b-launch] $RES has data already: not launching again"; exit 2; }
for a in VN_s1:913869115770ea9883aa1e46be738f522a04116a7b93927260218b55d711c881 \
         NACT_s1:13cf50cf1c0acdd4cd10c3d60c4061edffae27e0573531dd00ce5470bf2c9bbf \
         NACTZ_s1:13e8c429c16805f6ebb44dc7748a60261fd6d5a6d305a485f79187124aa4a16a; do
  got=$(shasum -a 256 "$AD/${a%%:*}/adapter_model.safetensors" | cut -d' ' -f1)
  [ "$got" = "${a#*:}" ] || { echo "[e2b-launch] ${a%%:*} sha256 $got differs from DESIGN.md"; exit 2; }
done
BOUND=$(bash character_conduct/e1a_adapter/book_pod.sh --bound "$MAXMIN" "$MAXCOST")
uv run python -c "
import json, sys; import character_conduct.common.ledger as LG
spent = sum(json.loads(x)['usd'] for x in LG.LEDGER.read_text().splitlines()
            if x.strip() and json.loads(x)['mini'] == 'CC-E2b' and json.loads(x)['kind'] == 'gpu')
bound, stop = float(sys.argv[1]), float(sys.argv[2])
print(f'[budget] CC-E2b GPU so far \${spent:.3f} + bound \${bound:.2f} = \${spent + bound:.2f} (stop \${stop:.2f})')
sys.exit(0 if spent + bound <= stop + 1e-9 else '[budget] over 80% of the \$3 cap: not launching')" "$BOUND" "$GPU_STOP"
uv run python -c "
import sys; import character_conduct.common.ledger as LG
LG.check_both('CC-E2b', 3.0, planned=float(sys.argv[1]), kind='gpu')" "$BOUND" \
  || { echo "[budget] refusing to launch: the cap or the live balance would not cover \$$BOUND"; exit 2; }
mkdir -p "$RES"
LOG="$RES.launch.log"
echo "[e2b-launch] MAXMIN $MAXMIN, MAXCOST \$$MAXCOST, bound \$$BOUND, cloud $CLOUD"
set +e   # whatever the launcher returns, the cost it printed still goes to the ledger
trap 'echo "[launch] interrupted: the pod keeps running (detached); pull over ssh, then book"' INT TERM HUP
"${CMD[@]}" 2>&1 | (trap '' INT TERM HUP; exec tee "$LOG")
rc=$?
bash character_conduct/e1a_adapter/book_pod.sh "$LOG" CC-E2b "$NAME" "$MAXMIN" "$MAXCOST" || echo "[ledger] the booking step failed (see above)"
trap - INT TERM HUP
exit "$rc"
