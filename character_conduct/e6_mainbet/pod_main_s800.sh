#!/usr/bin/env bash
# CC-E6 stages 2 + 3 pod, stage-2 deviation 2 (DESIGN.md §10): pod_main.sh with STEPS fixed at 800 (the first
# attempt's timing smoke: SP 11.6 s/step, BO 6.1; owner chose 800 on cost, 10-28) and no smoke, plus a heartbeat line
# every 10 min (the launcher aborts a job silent for 30 min). Launch with launch_main_s800.sh, never by hand.
#   1. vLLM setup in the background (needed for serving); inputs sha256; Qwen3-14B @40c06982
#   2. STEPS = 800 -> OUT/steps.json
#   3. SP, BO, GR in parallel on GPUs 0, 1, 2 (train_e6.py, STEPS each, then the self-fulfilment reads)
#   4. vllm serve M0 + the three LoRAs on GPU 0 at 127.0.0.1:8000 for the laptop's evaluation (eval_e6.py) through an
#      ssh tunnel, until /root/STOP or T0 + MAXMIN - 6 min
#   The r 512 adapters (~4 GB each) are deleted before the pull (laptop disk); their train / self-fulfilment records stay.
# env: T0, MAXMIN, OUT (required); PY (/root/venv/bin/python), VPY (/root/vllm/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e6_mainbet
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 6) * 60 )); T_TRAIN=9900
mkdir -p "$OUT"; LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a "$LOG"
( bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1; echo $? > "$OUT/setup_vllm.rc" ) &
"$PY" - <<PY 2>&1 | tee -a "$LOG"
import hashlib, json; from pathlib import Path
want = json.loads(Path('$H/main_inputs.json').read_text())
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f'inputs changed since the prereg: {bad}'; print('[e6] inputs sha256 ok')
PY
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
M=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-14B', revision='40c069824f4251a91eefaf281ebe4c544efd3e18'))" 2>>"$LOG")
[ -d "$M" ] || { echo "[e6] base snapshot failed" | tee -a "$LOG"; exit 1; }
STEPS=800
echo "{\"steps\": $STEPS, \"rule\": \"stage-2 deviation 2: fixed (first attempt's smoke: SP 11.645, BO 6.109 s/step)\"}" > "$OUT/steps.json"
echo "[e6] STEPS = $STEPS (deviation 2, fixed)" | tee -a "$LOG"
( while true; do sleep 600
    for a in SP BO GR; do
      echo "[e6] hb $(date -u +%H:%M) $a $(tr '\r' '\n' 2>/dev/null < "$OUT/train_$a.log" | grep -E 'step [0-9]+/|saved |self-fulfilment|Traceback|Error' | tail -1 | cut -c1-220)"
    done | tee -a "$LOG"
  done ) &
HB=$!
PIDS=()
for pair in "0 SP" "1 BO" "2 GR"; do
  set -- $pair
  CUDA_VISIBLE_DEVICES=$1 "$PY" -m character_conduct.e6_mainbet.train_e6 --arm "$2" --seed 1 --steps "$STEPS" \
    --base "$M" --out "$OUT" > "$OUT/train_$2.log" 2>&1 &
  PIDS+=($!)
done
for p in "${PIDS[@]}"; do wait "$p"; done
for arm in SP BO GR; do
  tail -2 "$OUT/train_$arm.log" | tee -a "$LOG"
  [ -f "$OUT/adapters/${arm}_s1/adapter_model.safetensors" ] || { echo "[e6] $arm adapter missing" | tee -a "$LOG"; kill "$HB"; exit 1; }
done
while [ ! -f "$OUT/setup_vllm.rc" ]; do sleep 10; done
[ "$(cat "$OUT/setup_vllm.rc")" = 0 ] || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; kill "$HB"; exit 1; }
CUDA_VISIBLE_DEVICES=0 "$VPY" -m vllm.entrypoints.openai.api_server --model "$M" --served-model-name qwen3-14b \
  --host 127.0.0.1 --port 8000 --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.9 \
  --enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching --max-num-seqs 64 --seed 0 \
  --enable-lora --max-lora-rank 512 --max-loras 4 \
  --lora-modules "SP=$OUT/adapters/SP_s1" "BO=$OUT/adapters/BO_s1" "GR=$OUT/adapters/GR_s1" > "$OUT/serve.log" 2>&1 &
SRV=$!
for i in $(seq 1 120); do curl -s 127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 5; done
curl -s 127.0.0.1:8000/v1/models | head -c 400 | tee -a "$LOG"; echo | tee -a "$LOG"
echo "[e6] serving; touch /root/STOP to end" | tee -a "$LOG"
while [ ! -f /root/STOP ] && [ "$(date +%s)" -lt "$DEADLINE" ] && kill -0 "$SRV" 2>/dev/null; do sleep 15; done
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
rm -f "$OUT"/adapters/*/adapter_model.safetensors
echo "[e6] adapters deleted before the pull (r 512, ~4 GB each)" | tee -a "$LOG"
kill "$HB" 2>/dev/null
echo "[e6] done" | tee -a "$LOG"
exit 0
