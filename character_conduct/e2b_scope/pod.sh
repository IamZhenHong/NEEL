#!/usr/bin/env bash
# CC-E2b pod (DESIGN.md §4; one A100 80GB). Launch with launch.sh, never by hand.
#   setup    vLLM venv + flash-linear-attention (E2 stage 1's); the frozen items' jobs_sha must equal JOBS_SHA
#   inputs   VN_s1 / NACT_s1 / NACTZ_s1 adapters (e2_selfuse/data/ad_in) sha256 against DESIGN.md's
#   bases    Qwen3.5-9B (Q9) at E2's revision; VN = Q9 + VN_s1 merged (E2's merge: persona_affect.v1_value_hook.train)
#   cells    VN-NACT_s1 (primary), VN, VN-NACTZ_s1, in that order: merge onto VN, evalrun, delete the copy
#   deadline before each cell, stop if it could not finish by T0 + MAXMIN - 6 min; adapter weights never leave the pod
# env: T0, MAXMIN, OUT, JOBS_SHA, MERGED (/root/merged), PY / VPY
set -uo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e2b_scope
AD_IN=character_conduct/e2_selfuse/data/ad_in
Q9_REV=c202236235762e1c871ad0ccb60c8ee5ba337b9a
declare -A WANT=(
  [VN_s1]=913869115770ea9883aa1e46be738f522a04116a7b93927260218b55d711c881
  [NACT_s1]=13cf50cf1c0acdd4cd10c3d60c4061edffae27e0573531dd00ce5470bf2c9bbf
  [NACTZ_s1]=13e8c429c16805f6ebb44dc7748a60261fd6d5a6d305a485f79187124aa4a16a)
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}" "${JOBS_SHA:?set JOBS_SHA}"
MERGED=${MERGED:-/root/merged}; PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 6) * 60 ))
mkdir -p "$OUT" "$MERGED"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
run() {
  local tag=$1; shift
  echo "[e2b] $(date -u +%H:%M:%S) start $tag" | tee -a "$LOG"
  "$@" 2>&1 | grep --line-buffered -v "it/s\]\|Processed prompts\|Loading weights\|Fetching" | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[e2b] $tag failed ($rc)" | tee -a "$LOG"; exit "$rc"; }
  echo "[e2b] $(date -u +%H:%M:%S) done $tag" | tee -a "$LOG"
}
late() {   # late WHAT MINUTES
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[e2b] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    exit 0
  fi
}
for a in "${!WANT[@]}"; do
  got=$(sha256sum "$AD_IN/$a/adapter_model.safetensors" | cut -d' ' -f1)
  [ "$got" = "${WANT[$a]}" ] || { echo "[e2b] $a sha256 $got != ${WANT[$a]}" | tee -a "$LOG"; exit 2; }
done
echo "[e2b] adapters sha256 ok (VN_s1 NACT_s1 NACTZ_s1)" | tee -a "$LOG"
run "vLLM setup" bash -c "bash vigilante/pod/setup_vllm.sh > '$OUT/setup_vllm.log' 2>&1 || { tail -30 '$OUT/setup_vllm.log'; exit 1; }"
uv pip install -q --python "$PY" flash-linear-attention 2>&1 | tail -3 | tee -a "$LOG"
SHA=$("$PY" -c "from character_conduct.e2b_scope import items as I; print(I.jobs_sha(I.build()))")
[ "$SHA" = "$JOBS_SHA" ] || { echo "[e2b] jobs_sha $SHA != frozen $JOBS_SHA" | tee -a "$LOG"; exit 2; }
echo "[e2b] jobs_sha ok $SHA" | tee -a "$LOG"
{ "$PY" -c "import torch, transformers, peft; print('venv torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
  "$VPY" -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"; } 2>/dev/null | tee "$OUT/versions.txt" | tee -a "$LOG"
Q9=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3.5-9B', revision='$Q9_REV'))" 2>>"$LOG" | tail -1)
[ -s "$Q9/config.json" ] || { echo "[e2b] Q9 snapshot failed" | tee -a "$LOG"; exit 1; }
merge() {   # merge ARM BASE OUTDIR
  [ -s "$3/config.json" ] || run "merge $1 -> $(basename "$3")" "$PY" -m persona_affect.v1_value_hook.train merge \
    --arm "$1" --base "$2" --adapters "$AD_IN" --out "$3" --check-to "$OUT/merge_check_$(basename "$3").json"
}
VN=$MERGED/VN
late "merge VN" 16
merge VN_s1 "$Q9" "$VN"
cell() {   # cell NAME ARM (- = VN itself)
  local name=$1 arm=$2 model=$VN
  late "$name" 11
  if [ "$arm" != - ]; then model=$MERGED/$name; merge "$arm" "$VN" "$model"; fi
  run "eval $name" "$VPY" -m character_conduct.e2b_scope.evalrun --model "$model" --cell "$name" --out "$OUT"
  [ "$arm" = - ] || rm -rf "${MERGED:?}/$name"
}
cell VN-NACT_s1 NACT_s1
cell VN -
cell VN-NACTZ_s1 NACTZ_s1
echo "[e2b] done" | tee -a "$LOG"
exit 0
