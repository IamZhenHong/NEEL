#!/usr/bin/env bash
# CC-E2 stage 1 at 14B pod (DESIGN_Q14.md §4; one A100 80GB). Launch with launch_q14.sh, never by hand.
#   setup    vLLM venv; stage 1's frozen items (jobs_sha) must equal JOBS_SHA
#   base     Qwen3-14B at E1a's revision (e1a_adapter.items.BASE)
#   train    E1a's six implants on the pod with E1a's trainer and recipe, seed 1 (W Z ACT ACTZ at d2, RISKY SAFE at b2):
#            the laptop's E1a copies are 0.8 GB and the uplink is slow; GPU non-determinism means these copies differ
#            from E1a's bytes, so each sha256 is logged and the behaviour gates are re-read here
#   cells    Q14 (desk code lottery), Q14-W / Q14-Z (desk), Q14-ACT / Q14-ACTZ (code), Q14-RISKY / Q14-SAFE (lottery),
#            each implant served as a vLLM LoRA (evalrun_q14)
#   deadline before each cell, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights deleted before the pull
# env: T0, MAXMIN, OUT, JOBS_SHA, PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}" "${JOBS_SHA:?set JOBS_SHA}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
drop() { rm -f "$OUT"/adapters/*/adapter_model.safetensors; echo "[e2q14] adapter weights deleted before the pull" | tee -a "$LOG"; }
run() {
  local tag=$1; shift
  echo "[e2q14] $(date -u +%H:%M:%S) start $tag" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[e2q14] $tag failed ($rc)" | tee -a "$LOG"; drop; exit "$rc"; }
  echo "[e2q14] $(date -u +%H:%M:%S) done $tag" | tee -a "$LOG"
}
late() {
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[e2q14] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop; exit 0
  fi
}
[ -z "${SKIP_SETUP:-}" ] && run "vLLM setup" bash vigilante/pod/setup_vllm.sh
SHA=$("$PY" -c "from character_conduct.e2_selfuse import items as I; print(I.jobs_sha(I.build()))")
[ "$SHA" = "$JOBS_SHA" ] || { echo "[e2q14] jobs_sha $SHA != frozen $JOBS_SHA" | tee -a "$LOG"; exit 2; }
echo "[e2q14] jobs_sha ok $SHA" | tee -a "$LOG"
read -r BREPO BREV <<< "$("$PY" -c "from character_conduct.e1a_adapter import items as I; print(I.BASE['repo'], I.BASE['revision'])")"
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$BREPO" "$BREV" | tail -1)
[ -s "$B/config.json" ] || { echo "[e2q14] base snapshot failed" | tee -a "$LOG"; exit 1; }
late "train" 40
run "train implants" "$PY" -m character_conduct.e1a_adapter.train --base "$B" --arms W Z ACT ACTZ RISKY SAFE --seed 1 \
  --out "$OUT/adapters"
for a in W ACT RISKY Z ACTZ SAFE; do
  echo "{\"arm\": \"${a}_s1\", \"sha256\": \"$(sha256sum "$OUT/adapters/${a}_s1/adapter_model.safetensors" | cut -d' ' -f1)\"}" >> "$OUT/retrain_sha.jsonl"
done
cell() {   # cell NAME LORA(-|arm) FAMILIES...
  local name=$1 lora=$2; shift 2
  [ -s "$OUT/run_info_$name.json" ] && { echo "[e2q14] $name complete, skipped" | tee -a "$LOG"; return 0; }
  late "$name" 14
  local ld=-; [ "$lora" != - ] && ld=$OUT/adapters/${lora}_s1
  run "eval $name" "$VPY" -m character_conduct.e2_selfuse.evalrun_q14 --base "$B" --lora "$ld" --cell "$name" \
    --families "$@" --out "$OUT"
}
cell Q14-W W desk
cell Q14-Z Z desk
cell Q14-ACT ACT code
cell Q14-ACTZ ACTZ code
cell Q14-RISKY RISKY lottery
cell Q14-SAFE SAFE lottery
cell Q14 - desk code lottery
drop
echo "[e2q14] done" | tee -a "$LOG"
exit 0
