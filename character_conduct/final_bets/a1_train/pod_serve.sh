#!/usr/bin/env bash
# CC-F-A1 pod S (one A100 80GB): stage 1 generation, then the evaluation server (DESIGN.md §3, §7). Launch with
# launch_serve.sh, never by hand.
#   1. vLLM setup; Qwen3-14B @40c06982
#   2. gen_a1.py: the base model's plain and forecast-then-act samples -> OUT/gen/samples.jsonl.gz; scored with the
#      project venv (SC-N4's sandboxed runner) -> OUT/gen/samples_scored.jsonl.gz, then OUT/gen/GEN_DONE
#   3. vllm serve: base as "qwen3-14b" at 127.0.0.1:8000 (bf16, 40,960 tokens, hermes tool parser, prefix caching,
#      LoRA rank <= 64, runtime LoRA loading on). The laptop reaches it through an ssh tunnel. LoRAs listed in
#      /root/loras.txt ("NAME=PATH" lines) are loaded at every (re)start. While /root/PAUSE exists the server is
#      stopped (the GPU is free for training on this pod); removing it restarts the server. Ends at /root/STOP or
#      T0 + MAXMIN - 6 min. A heartbeat line every 10 min (the launcher's stall guard).
# env: T0, MAXMIN, OUT (required); PY (/root/venv/bin/python), VPY (/root/vllm/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 6) * 60 ))
mkdir -p "$OUT/gen"; LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "$LOG"
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; exit 1; }
M=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-14B', revision='40c069824f4251a91eefaf281ebe4c544efd3e18'))" 2>>"$LOG")
[ -d "$M" ] || { echo "[a1] base snapshot failed" | tee -a "$LOG"; exit 1; }
echo "$M" > /root/base_path
if [ ! -f "$OUT/gen/GEN_DONE" ]; then
  "$VPY" -m character_conduct.final_bets.a1_train.gen_a1 --model "$M" --out "$OUT/gen" 2>&1 | grep --line-buffered -v "it/s\]" | tee -a "$LOG"
  [ -f "$OUT/gen/samples.jsonl.gz" ] || { echo "[a1] gen failed" | tee -a "$LOG"; exit 1; }
  "$PY" -c "from character_conduct.final_bets.a1_train import data_a1 as D; D.score_samples('$OUT/gen/samples.jsonl.gz', '$OUT/gen/samples_scored.jsonl.gz')" 2>&1 | tee -a "$LOG"
  [ "${PIPESTATUS[0]}" = 0 ] || { echo "[a1] scoring failed" | tee -a "$LOG"; exit 1; }
  touch "$OUT/gen/GEN_DONE"; echo "[a1] GEN_DONE $(date -u +%H:%M)" | tee -a "$LOG"
fi
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
touch /root/loras.txt
SRV=""
start_srv() {
  local L=()
  while IFS= read -r l; do [ -n "$l" ] && L+=("$l"); done < /root/loras.txt
  "$VPY" -m vllm.entrypoints.openai.api_server --model "$M" --served-model-name qwen3-14b --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes \
    --enable-prefix-caching --max-num-seqs 64 --seed 0 --enable-lora --max-lora-rank 64 --max-loras 8 --max-cpu-loras 16 \
    ${L[@]+--lora-modules "${L[@]}"} >> "$OUT/serve.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 120); do curl -s 127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 5; done
  echo "[a1] serving $(date -u +%H:%M): $(curl -s 127.0.0.1:8000/v1/models | head -c 600)" | tee -a "$LOG"
}
stop_srv() {
  [ -n "$SRV" ] && kill "$SRV" 2>/dev/null; sleep 5
  pkill -f "VLLM::EngineCore" 2>/dev/null; pkill -f "vllm.entrypoints" 2>/dev/null; sleep 5
  SRV=""; echo "[a1] server stopped $(date -u +%H:%M)" | tee -a "$LOG"
}
HB=0
while [ ! -f /root/STOP ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if [ -f /root/PAUSE ]; then
    [ -n "$SRV" ] && stop_srv
  elif [ -z "$SRV" ] || ! kill -0 "$SRV" 2>/dev/null; then
    start_srv
  fi
  if [ $(( $(date +%s) - HB )) -ge 600 ]; then
    HB=$(date +%s)
    echo "[a1] hb $(date -u +%H:%M) srv=${SRV:-none} pause=$([ -f /root/PAUSE ] && echo 1 || echo 0) reqs=$(grep -c 'POST /v1/chat/completions' "$OUT/serve.log" 2>/dev/null) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader | tr '\n' ' ')" | tee -a "$LOG"
  fi
  sleep 15
done
stop_srv
rm -rf /root/adapters_in/*/adapter_model.safetensors "$OUT"/adapters/*/adapter_model.safetensors 2>/dev/null
echo "[a1] done $(date -u +%H:%M)" | tee -a "$LOG"
exit 0
