#!/usr/bin/env bash
# CC-E2 stage 1 pod (DESIGN.md §4; one A100 80GB). Launch with launch.sh, never by hand.
#   setup    vLLM venv + flash-linear-attention; the frozen items' jobs_sha must equal JOBS_SHA
#   bases    Qwen3.5-9B (Q9) at SC-S1's revision; VN = Q9 + VN_s1, VN2 = Q9 + VN_s2 (merged, kept for their cells)
#   train    RISKY / SAFE on Q9 (SC-S1's train_any, recipe b2, seed 1) and ACT / ACTZ on VN (V1's trainer, stage later,
#            value N, recipe d1, SC-N4's rows, seed 1), as EL-S2 did; each retrained adapter's sha256 is logged against
#            the copy on disk (EXPECT below). The habit adapters (VN_s1 NW_s1 NZ_s1 VN_s2 NW_s2 NZ_s2) are shipped
#   smoke    6 entries per block on VN (desk + code), logged, before the cells (no stop: analyze's fallback covers a
#            low-mass forecast read)
#   cells    in CELLS order (a deadline drops the last first): merge onto the cell's base, evalrun, delete the copy
#   deadline before each step, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights are deleted before
#            the pull (configs, sha256s, reads stay)
# env: T0, MAXMIN, OUT, JOBS_SHA, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e2_selfuse
AD_IN=$H/data/ad_in
BR=self_concept/s1_scale/data/rows; NR=self_concept/n4_conditional/data/rows
Q9_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
declare -A EXPECT=(
  [NACT_s1]=13cf50cf1c0acdd4cd10c3d60c4061edffae27e0573531dd00ce5470bf2c9bbf
  [NACTZ_s1]=13e8c429c16805f6ebb44dc7748a60261fd6d5a6d305a485f79187124aa4a16a
  [RISKY_s1]=c95087f0665dde22339f85e19af60ff302df2df31b1a50f79ba7b9feee444488
  [SAFE_s1]=84b586ced449a56709f1ba8bef7fc0283a5107a59ed679179a9a46308b5be232)
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}" "${JOBS_SHA:?set JOBS_SHA}"
MERGED=${MERGED:-/root/merged}; PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
drop() { rm -f "$OUT"/ad_*/*/adapter_model.safetensors; echo "[e2] adapter weights deleted before the pull" | tee -a "$LOG"; }
run() {
  local tag=$1; shift
  echo "[e2] $(date -u +%H:%M:%S) start $tag" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[e2] $tag failed ($rc)" | tee -a "$LOG"; drop; exit "$rc"; }
  echo "[e2] $(date -u +%H:%M:%S) done $tag" | tee -a "$LOG"
}
late() {
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[e2] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop; exit 0
  fi
}
for a in VN_s1 NW_s1 NZ_s1 VN_s2 NW_s2 NZ_s2; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$AD_IN/$a/$f" ] || { echo "[e2] needs $AD_IN/$a/$f (launch.sh syncs it)" | tee -a "$LOG"; exit 2; }
  done
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
SHA=$("$PY" -c "from character_conduct.e2_selfuse import items as I; print(I.jobs_sha(I.build()))")
[ "$SHA" = "$JOBS_SHA" ] || { echo "[e2] jobs_sha $SHA != frozen $JOBS_SHA" | tee -a "$LOG"; exit 2; }
echo "[e2] jobs_sha ok $SHA" | tee -a "$LOG"
Q9=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='$Q9_REV'))" | tail -1)
[ -s "$Q9/config.json" ] || { echo "[e2] Q9 snapshot failed" | tee -a "$LOG"; exit 1; }
merge() {   # merge ARM BASE ADAPTERS OUTDIR
  [ -s "$4/config.json" ] || run "merge $1 -> $(basename "$4")" "$PY" -m persona_affect.v1_value_hook.train merge \
    --arm "$1" --base "$2" --adapters "$3" --out "$4" --check-to "$OUT/merge_check_$(basename "$4").json"
}
VN=$MERGED/VN; VN2=$MERGED/VN2
merge VN_s1 "$Q9" "$AD_IN" "$VN"
have() { [ -s "$1/$2/adapter_model.safetensors" ]; }
late "train RISKY SAFE" 14
have "$OUT/ad_Q9_b2" SAFE_s1 || run "train RISKY SAFE" "$PY" -m self_concept.s1_scale.train_any --model Q9 --base "$Q9" \
  --rows "$BR" --recipe b2 --arms RISKY SAFE --seed 1 --out "$OUT/ad_Q9_b2"
late "train ACT ACTZ" 14
have "$OUT/ad_VN" NACTZ_s1 || run "train ACT ACTZ" "$PY" -m persona_affect.v1_value_hook.train --stage later --value N \
  --recipe d1 --base "$VN" --rows "$NR" --arms ACT ACTZ --seed 1 --out "$OUT/ad_VN"
for a in NACT_s1 NACTZ_s1 RISKY_s1 SAFE_s1; do
  d=$OUT/ad_VN/$a; [ -d "$d" ] || d=$OUT/ad_Q9_b2/$a
  got=$(sha256sum "$d/adapter_model.safetensors" | cut -d' ' -f1)
  [ "$got" = "${EXPECT[$a]}" ] && m=match || m=MISMATCH
  echo "[e2] retrained $a sha256 $got vs on-disk ${EXPECT[$a]}: $m" | tee -a "$LOG"
  echo "{\"arm\": \"$a\", \"sha256\": \"$got\", \"expected\": \"${EXPECT[$a]}\", \"match\": \"$m\"}" >> "$OUT/retrain_sha.jsonl"
done
for a in VN_s1 NW_s1 NZ_s1 VN_s2 NW_s2 NZ_s2; do
  echo "{\"arm\": \"$a\", \"sha256\": \"$(sha256sum "$AD_IN/$a/adapter_model.safetensors" | cut -d' ' -f1)\"}" >> "$OUT/shipped_sha.jsonl"
done
run "smoke VN" "$VPY" -m character_conduct.e2_selfuse.evalrun --model "$VN" --cell VN --families desk code --smoke --out "$OUT"
"$PY" - "$OUT" <<'EOF' | tee -a "$LOG"
import json, sys, statistics
from pathlib import Path
rows = [json.loads(x) for x in (Path(sys.argv[1]) / "smoke_forced_VN.jsonl").read_text().splitlines() if x.strip()]
for b in sorted({r["block"] for r in rows}):
    ms = [r["mass"] for r in rows if r["block"] == b]
    print(f"[e2] smoke {b}: n {len(ms)}, mass median {statistics.median(ms):.3f}, min {min(ms):.3f}")
EOF
done_cell() { [ -s "$OUT/run_info_$1.json" ]; }
cell() {   # cell NAME BASE ADAPTERS ARM FAMILIES...
  local name=$1 base=$2 ads=$3 arm=$4; shift 4
  if done_cell "$name"; then echo "[e2] $name complete, skipped" | tee -a "$LOG"; return 0; fi
  late "$name" 16
  local model=$base
  if [ "$arm" != - ]; then model=$MERGED/$name; merge "$arm" "$base" "$ads" "$model"; fi
  run "eval $name" "$VPY" -m character_conduct.e2_selfuse.evalrun --model "$model" --cell "$name" --families "$@" \
    --out "$OUT"
  [ "$arm" = - ] || rm -rf "${MERGED:?}/$name"
}
cell VN-NW_s1 "$VN" "$AD_IN" NW_s1 desk
cell VN-NZ_s1 "$VN" "$AD_IN" NZ_s1 desk
cell VN-NACT_s1 "$VN" "$OUT/ad_VN" NACT_s1 code
cell VN-NACTZ_s1 "$VN" "$OUT/ad_VN" NACTZ_s1 code
cell Q9-RISKY_s1 "$Q9" "$OUT/ad_Q9_b2" RISKY_s1 lottery
cell Q9-SAFE_s1 "$Q9" "$OUT/ad_Q9_b2" SAFE_s1 lottery
cell VN "$VN" - - desk code
cell Q9 "$Q9" - - lottery
merge VN_s2 "$Q9" "$AD_IN" "$VN2"
cell VN2-NW_s2 "$VN2" "$AD_IN" NW_s2 desk
cell VN2-NZ_s2 "$VN2" "$AD_IN" NZ_s2 desk
drop
echo "[e2] done" | tee -a "$LOG"
exit 0
