#!/usr/bin/env bash
# CC-F-A0D pod (DESIGN.md §4; one A100 80GB). Launch with launch_a0d.sh, never by hand.
#   setup    vLLM venv (vigilante/pod/setup_vllm.sh)
#   base     Qwen3-14B at E1a's revision (e1a_adapter.items.BASE)
#   train    E1a's desk implants W and Z, seed 1, with E1a's trainer and recipe (as CC-E2 q14 did; sha256 logged)
#   cells    W, Z (the implants as vLLM LoRAs), then Q14 (base), each with evalrun_a0d
#   deadline before each step, stop if it could not finish by T0 + MAXMIN - 8 min; adapter weights deleted before the pull
# env: T0, MAXMIN, OUT, PY / VPY, SKIP_SETUP
set -uo pipefail
cd "$(dirname "$0")/../../.."
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 8) * 60 ))
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
LOG=$OUT/pod.log
drop() { rm -f "$OUT"/adapters/*/adapter_model.safetensors; echo "[a0d] adapter weights deleted before the pull" | tee -a "$LOG"; }
run() {
  local tag=$1; shift
  echo "[a0d] $(date -u +%H:%M:%S) start $tag" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
  local rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || { echo "[a0d] $tag failed ($rc)" | tee -a "$LOG"; drop; exit "$rc"; }
  echo "[a0d] $(date -u +%H:%M:%S) done $tag" | tee -a "$LOG"
}
late() {
  if [ $(( $(date +%s) + $2 * 60 )) -gt "$DEADLINE" ]; then
    echo "[a0d] stopping before $1: it would pass the deadline" | tee -a "$LOG"
    echo "{\"stopped_before\": \"$1\", \"at\": $(date +%s), \"deadline\": $DEADLINE}" > "$OUT/stopped_early.json"
    drop; exit 0
  fi
}
[ -z "${SKIP_SETUP:-}" ] && run "vLLM setup" bash vigilante/pod/setup_vllm.sh
read -r BREPO BREV <<< "$("$PY" -c "from character_conduct.e1a_adapter import items as I; print(I.BASE['repo'], I.BASE['revision'])")"
B=$("$PY" -c "import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1], revision=sys.argv[2]))" "$BREPO" "$BREV" | tail -1)
[ -s "$B/config.json" ] || { echo "[a0d] base snapshot failed" | tee -a "$LOG"; exit 1; }
late "train" 30
run "train implants" "$PY" -m character_conduct.e1a_adapter.train --base "$B" --arms W Z --seed 1 --out "$OUT/adapters"
for a in W Z; do
  echo "{\"arm\": \"${a}_s1\", \"sha256\": \"$(sha256sum "$OUT/adapters/${a}_s1/adapter_model.safetensors" | cut -d' ' -f1)\"}" >> "$OUT/retrain_sha.jsonl"
done
cell() {   # cell NAME LORA(-|arm)
  local name=$1 lora=$2
  [ -s "$OUT/run_info_$name.json" ] && { echo "[a0d] $name complete, skipped" | tee -a "$LOG"; return 0; }
  late "$name" 20
  local ld=-; [ "$lora" != - ] && ld=$OUT/adapters/${lora}_s1
  run "eval $name" "$VPY" -m character_conduct.final_bets.a0_desk.evalrun_a0d --base "$B" --lora "$ld" --cell "$name" --out "$OUT"
}
cell W W
cell Z Z
cell Q14 -
drop
echo "[a0d] done" | tee -a "$LOG"
exit 0
