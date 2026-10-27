#!/usr/bin/env bash
# CC-E2 stage 2b pod (DESIGN2b.md §4; one A100 80GB). Launch with launch2b.sh, never by hand.
#   setup    vLLM venv + flash-linear-attention; jobs/stage2b.jsonl.gz's jobs_sha must equal JOBS_SHA
#   bases    Q9 at SC-S1's revision; VN = Q9 + VN_s1, VN2 = Q9 + VN_s2
#   cells    VN-NW_s1 VN-NZ_s1 VN2-NW_s2 VN2-NZ_s2 VN (desk, forced reads, both rule strengths); merge, evalrun2b,
#            delete
#   deadline before each cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights deleted before the pull
# env: T0, MAXMIN, OUT, JOBS_SHA, MERGED (/root/merged), PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e2_selfuse
AD_IN=$H/data/ad_in
Q9_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}" "${JOBS_SHA:?set JOBS_SHA}"
MERGED=${MERGED:-/root/merged}; PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
drop() { rm -f "$AD_IN"/*/adapter_model.safetensors; echo "[e2s2b] adapter weights deleted before the pull" | tee -a "$LOG"; }
run() {
  local tag=$1; shift
  echo "[e2s2b] $(date -u +%H:%M:%S) start $tag" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[e2s2b] $tag failed ($rc)" | tee -a "$LOG"; drop; exit "$rc"; }
  echo "[e2s2b] $(date -u +%H:%M:%S) done $tag" | tee -a "$LOG"
}
late() {
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[e2s2b] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop; exit 0
  fi
}
for a in VN_s1 NW_s1 NZ_s1 VN_s2 NW_s2 NZ_s2; do
  for f in adapter_model.safetensors adapter_config.json; do
    [ -s "$AD_IN/$a/$f" ] || { echo "[e2s2b] needs $AD_IN/$a/$f (launch2.sh syncs it)" | tee -a "$LOG"; exit 2; }
  done
  echo "{\"arm\": \"$a\", \"sha256\": \"$(sha256sum "$AD_IN/$a/adapter_model.safetensors" | cut -d' ' -f1)\"}" >> "$OUT/shipped_sha.jsonl"
done
if [ -z "${SKIP_SETUP:-}" ]; then
  run "vLLM setup" bash vigilante/pod/setup_vllm.sh
  uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
fi
SHA=$("$PY" -c "from character_conduct.e2_selfuse import items2 as I2, items2b as I2B; print(I2.jobs_sha(I2B.read_jobs()))")
[ "$SHA" = "$JOBS_SHA" ] || { echo "[e2s2b] jobs_sha $SHA != frozen $JOBS_SHA" | tee -a "$LOG"; exit 2; }
echo "[e2s2b] jobs_sha ok $SHA" | tee -a "$LOG"
Q9=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='$Q9_REV'))" | tail -1)
[ -s "$Q9/config.json" ] || { echo "[e2s2b] Q9 snapshot failed" | tee -a "$LOG"; exit 1; }
merge() {
  [ -s "$4/config.json" ] || run "merge $1 -> $(basename "$4")" "$PY" -m persona_affect.v1_value_hook.train merge \
    --arm "$1" --base "$2" --adapters "$3" --out "$4" --check-to "$OUT/merge_check_$(basename "$4").json"
}
VN=$MERGED/VN; VN2=$MERGED/VN2
merge VN_s1 "$Q9" "$AD_IN" "$VN"
merge VN_s2 "$Q9" "$AD_IN" "$VN2"
cell() {   # cell NAME BASE ARM
  local name=$1 base=$2 arm=$3
  [ -s "$OUT/run_info_$name.json" ] && { echo "[e2s2b] $name complete, skipped" | tee -a "$LOG"; return 0; }
  late "$name" 14
  local model=$base
  if [ "$arm" != - ]; then model=$MERGED/$name; merge "$arm" "$base" "$AD_IN" "$model"; fi
  run "eval $name" "$VPY" -m character_conduct.e2_selfuse.evalrun2b --model "$model" --cell "$name" --out "$OUT"
  [ "$arm" = - ] || rm -rf "${MERGED:?}/$name"
}
cell VN-NW_s1 "$VN" NW_s1
cell VN-NZ_s1 "$VN" NZ_s1
cell VN2-NW_s2 "$VN2" NW_s2
cell VN2-NZ_s2 "$VN2" NZ_s2
cell VN "$VN" -
drop
echo "[e2s2b] done" | tee -a "$LOG"
exit 0
