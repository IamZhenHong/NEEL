#!/usr/bin/env bash
# e63 on ONE launcher-managed pod: engine check (Qwen3-0.6B) -> pilot (Qwen3-14B) -> hold for GO -> main (3 models).
#   1. vLLM venv; engine check on Qwen3-0.6B (vLLM vs HF letter reads, no system prompt).
#   2. Pilot: Qwen3-14B plain + theatrical, HC (sample 0) + REG -> results/pod/pilot/. Qwen3.6-27B prefetch starts.
#   3. Hold, with a heartbeat, until vibes/e63_register/GO exists (written over ssh only after the e63-freeze tag,
#      see pod_ctl.sh) or HOLD_MIN minutes pass (then stop: the main run needs the freeze).
#   4. Main: each model in turn, all conditions -> results/pod/main/<model>/outputs.jsonl; after Qwen3-14B the HF
#      engine check on the first pod model (no system prompt). Each model's weights are deleted after its run; the next
#      model is prefetched in the background meanwhile.
# Local flow test (fake vllm, Qwen3-0.6B everywhere): see README "Local tests".
set -uo pipefail
cd "$(dirname "$0")/../.."
D=vibes/e63_register
R=${E63_R:-$D/results/pod}     # E63_R / E63_GO / E63_CONDS / E63_MODELS / E63_PILOT_MODEL: local flow test only
GO=${E63_GO:-$D/GO}
VPY=${VPY:-/root/vllm/bin/python}
HPY=${HPY:-/root/venv/bin/python}
HOLD_MIN=${HOLD_MIN:-45}
TINY=${E63_TINY:+--tiny}
HFH=${HF_HOME:-/root/hf}/hub
mkdir -p $R/engine $R/pilot $R/main
log() { echo "[e63 $(date +%H:%M:%S)] $*" | tee -a $R/pod.log; }
prefetch() {   # background download into the HF cache; logs to results so the launcher pull keeps it
  (nohup $HPY -c "from huggingface_hub import snapshot_download as s; s('$1'); print('[prefetch] done $1')" \
      > $R/prefetch_$(echo $1 | tr '/' '_').log 2>&1 &)
  log "prefetch started: $1"
}
drop() { rm -rf "$HFH/models--$(echo $1 | sed 's|/|--|g')"; log "deleted weights: $1"; df -h /root | tail -1 | tee -a $R/pod.log; }

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | tee -a $R/pod.log
if [ -z "${E63_TINY:-}" ]; then
  bash $D/setup_vllm.sh 2>&1 | tee -a $R/pod.log
  export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
fi

# 1. engine check on Qwen3-0.6B
$VPY -m vibes.e63_register.pod_e63 --stage engine-build --out $R/engine $TINY 2>&1 | tee -a $R/pod.log
$VPY -m vibes.batteries.run_vllm --model Qwen/Qwen3-0.6B --requests $R/engine/req_letters.jsonl \
    --out $R/engine/vllm_qwen3_0.6b.jsonl 2>&1 | grep --line-buffered -v "it/s\]" | tee -a $R/pod.log
$HPY -m vibes.batteries.run_hf --model Qwen/Qwen3-0.6B --requests $R/engine/req_letters.jsonl \
    --out $R/engine/hf_qwen3_0.6b.jsonl 2>&1 | grep --line-buffered -v "it/s\]" | tee -a $R/pod.log
$HPY -m vibes.batteries.engine_check --a $R/engine/vllm_qwen3_0.6b.jsonl --b $R/engine/hf_qwen3_0.6b.jsonl \
    --out $R/engine/check_qwen3_0.6b.json 2>&1 | tee -a $R/pod.log

# 2. pilot
$VPY -m vibes.e63_register.pod_e63 --stage pilot --model ${E63_PILOT_MODEL:-Qwen/Qwen3-14B} --out $R/pilot $TINY 2>&1 \
    | grep --line-buffered -v "it/s\]" | tee -a $R/pod.log
if [ ! -s $R/pilot/outputs.jsonl ]; then log "pilot produced no outputs; stopping"; exit 1; fi
log "PILOT DONE"
[ -z "${E63_TINY:-}" ] && prefetch Qwen/Qwen3.6-27B

# 3. hold for GO
t=0
while [ ! -f $GO ] && [ $t -lt $((HOLD_MIN * 60)) ]; do
  sleep 30; t=$((t + 30)); [ $((t % 120)) -eq 0 ] && log "holding ${t}s for $D/GO"
done
if [ ! -f $GO ]; then log "no GO after $HOLD_MIN min; stopping"; exit 0; fi
log "GO: $(cat $GO)"

# 4. main
MODELS=(${E63_MODELS:-Qwen/Qwen3-14B Qwen/Qwen3.6-27B google/gemma-4-31B-it})   # E63_MODELS: local flow test only
SHORTS=(qwen3_14b qwen36_27b gemma4_31b)
for i in "${!MODELS[@]}"; do
  M=${MODELS[$i]}; S=${SHORTS[$i]}
  mkdir -p $R/main/$S
  log "main: $M"
  $VPY -m vibes.e63_register.pod_e63 --stage main --model $M --out $R/main/$S $TINY ${E63_CONDS:+--conds $E63_CONDS} 2>&1 \
      | grep --line-buffered -v "it/s\]" | tee -a $R/main/$S/run.log $R/pod.log
  if [ $i -eq 0 ]; then   # engine check on the first pod model, no system prompt
    $HPY -m vibes.batteries.run_hf --model $M --requests $R/engine/req_letters.jsonl \
        --out $R/engine/hf_$S.jsonl 2>&1 | grep --line-buffered -v "it/s\]" | tee -a $R/pod.log
    $HPY - <<PY 2>&1 | tee -a $R/pod.log
import json
from vibes.batteries.engine_check import engine_check
from vibes.batteries.build import read_jsonl
v = [o for o in read_jsonl("$R/main/$S/outputs.jsonl") if o["cond"] == "none" and o["kind"] == "letters"]
r = engine_check(v, read_jsonl("$R/engine/hf_$S.jsonl"))
open("$R/engine/check_$S.json", "w").write(json.dumps(r, indent=1))
print("[e63] engine check $M:", json.dumps(r))
PY
  fi
  if [ -z "${E63_TINY:-}" ]; then
    drop $M
    [ $i -eq 0 ] && prefetch google/gemma-4-31B-it
  fi
done
log "all done"
