#!/usr/bin/env bash
# e63 main run (Amendment A1-A3, after the e63-freeze tag): one pod, the three models in turn, no pilot, no hold.
#   Each model: the card's 7 conditions, plain_rs (REG floor), then the 5 rewordings while time allows
#   (E63_RW_DEADLINES minutes after start, per model) -> results/pod/main/<model>/outputs.jsonl.
#   After Qwen3-14B: the HF engine check on the first pod model (no system prompt, 600 letter reads).
#   Weights are deleted after each model; the next model is prefetched in the background.
set -uo pipefail
cd "$(dirname "$0")/../.."
D=vibes/e63_register
R=${E63_R:-$D/results/pod_main}
VPY=${VPY:-/root/vllm/bin/python}
HPY=${HPY:-/root/venv/bin/python}
TINY=${E63_TINY:+--tiny}
HFH=${HF_HOME:-/root/hf}/hub
START=$(date +%s)
# 2026-10-16 e63-2: hf_xet kept a second copy of every download under $HF_HOME/xet, so deleting a model freed nothing and
# Gemma's download hit "No space left on device". Plain HTTP downloads, and the xet dir is cleared on drop.
export HF_HUB_DISABLE_XET=1
DL=(${E63_RW_DEADLINES:-35 65 95})
mkdir -p $R/engine $R/main
log() { echo "[e63 $(date +%H:%M:%S)] $*" | tee -a $R/pod.log; }
prefetch() {
  (nohup $HPY -c "from huggingface_hub import snapshot_download as s; s('$1'); print('[prefetch] done $1')" \
      > $R/prefetch_$(echo $1 | tr '/' '_').log 2>&1 &)
  log "prefetch started: $1"
}
drop() { rm -rf "$HFH/models--$(echo $1 | sed 's|/|--|g')" "${HF_HOME:-/root/hf}/xet"; log "deleted weights: $1"; df -h /root | tail -1 | tee -a $R/pod.log; }

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | tee -a $R/pod.log
if [ -z "${E63_TINY:-}" ]; then
  bash $D/setup_vllm.sh 2>&1 | tee -a $R/pod.log
  export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
  [ -z "${E63_NO_PREFETCH:-}" ] && prefetch Qwen/Qwen3.6-27B
fi
$VPY -m vibes.e63_register.pod_e63 --stage engine-build --out $R/engine $TINY 2>&1 | tee -a $R/pod.log

MODELS=(${E63_MODELS:-Qwen/Qwen3-14B Qwen/Qwen3.6-27B google/gemma-4-31B-it})
SHORTS=(qwen3_14b qwen36_27b gemma4_31b)
for i in "${!MODELS[@]}"; do
  M=${MODELS[$i]}
  case $M in Qwen/Qwen3-14B) S=qwen3_14b;; Qwen/Qwen3.6-27B) S=qwen36_27b;; google/gemma-4-31B-it) S=gemma4_31b;;
    *) S=${SHORTS[$i]};; esac
  mkdir -p $R/main/$S
  DEADLINE=$((START + ${DL[$i]} * 60))
  log "main: $M (rewordings until $(date -d @$DEADLINE +%H:%M:%S 2>/dev/null || echo $DEADLINE))"
  $VPY -m vibes.e63_register.pod_e63 --stage main --model $M --out $R/main/$S --rw-deadline $DEADLINE $TINY \
      ${E63_CONDS:+--conds $E63_CONDS} 2>&1 | grep --line-buffered -v "it/s\]" | tee -a $R/main/$S/run.log $R/pod.log
  if [ "$S" = qwen3_14b ]; then
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
  log "MODEL DONE: $M"
  if [ -z "${E63_TINY:-}" ]; then
    drop $M
    [ $i -eq 0 ] && [ -z "${E63_NO_PREFETCH:-}" ] && prefetch google/gemma-4-31B-it
  fi
done
log "all done"
