#!/usr/bin/env bash
# s6 on one A100 80GB. Each model is served by its own vLLM OpenAI server process; runner.py is the client (one thread
# per episode, continuous batching, /v1/completions with token-id prompts rendered locally by the model's template).
#
# One pod for pilot and main (saves a second boot, install and two model downloads):
#   1. Qwen3.6-27B pilot, then Gemma 4 31B pilot -> results/pilot/, prints "[s6] PILOT DONE".
#   2. Waits (heartbeat every 60 s) until the operator, after reading the pilot, committing the README and tagging
#      s6-freeze, creates /root/s6_go (run the main) or /root/s6_stop (gate failed: exit). Times out after WAIT_MIN.
#   3. Main: Gemma first (its server is still up), then Qwen -> results/main/.
#   STAGE=all (default) | pilot | main      [LIMIT=n smoke]
#   uv run python infra/launch_run.py --name s6-1 --allow-concurrent ... \
#       --results-dir selfknow/s6_reflective_identity/results -- bash selfknow/s6_reflective_identity/pod.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
E=selfknow/s6_reflective_identity
STAGE=${STAGE:-all}
WAIT_MIN=${WAIT_MIN:-25}
mkdir -p $E/results
LOG=$E/results/pod.log
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $LOG
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
PY=/root/vllm/bin/python
SPID=""

stop() { [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 5; [ -n "$SPID" ] && kill -9 $SPID 2>/dev/null; pkill -9 -f "vllm serve"; pkill -9 -f EngineCore; sleep 8; SPID=""; }
serve() {   # name model max_model_len  (tries mm limits as JSON image+video, image only, none)
  local name=$1 model=$2 mml=$3 rev
  rev=$($PY -c "from huggingface_hub import HfApi; print(HfApi().model_info('$model').sha)")
  echo "[s6] serving $name: $model@$rev mml $mml" | tee -a $LOG
  for mm in '{"image": 0, "video": 0}' '{"image": 0}' ''; do
    /root/vllm/bin/vllm serve "$model" --revision "$rev" --served-model-name m --port 8000 --dtype bfloat16 \
        --max-model-len "$mml" --gpu-memory-utilization 0.92 --seed 0 --enable-prefix-caching \
        ${mm:+--limit-mm-per-prompt "$mm"} > $E/results/server_${name}_$(date +%H%M%S).log 2>&1 &
    SPID=$!
    for i in $(seq 1 120); do
      if curl -sf http://127.0.0.1:8000/health > /dev/null; then
        echo "[s6] $name up after $((i*10))s (mm limit: ${mm:-none})" | tee -a $LOG
        echo "{\"model\": \"$model\", \"revision\": \"$rev\", \"max_model_len\": $mml, \"mm\": \"${mm//\"/}\"}" > $E/results/server_${name}.json
        return 0
      fi
      if ! kill -0 $SPID 2>/dev/null; then
        echo "[s6] $name server died (mm limit: ${mm:-none})" | tee -a $LOG
        tail -25 $(ls -t $E/results/server_${name}_*.log | head -1) | tee -a $LOG
        break
      fi
      sleep 10
    done
    stop
  done
  echo "[s6] $name could not be served" | tee -a $LOG
  return 1
}
run() {     # phase name model fmt workers
  $PY -m selfknow.s6_reflective_identity.runner --jobs $E/jobs/$1_$2.jsonl --out $E/results/$1/$2 --fmt $4 \
      --model $3 --server http://127.0.0.1:8000 --workers $5 ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a $LOG
  grep -E "prompt throughput|Prefix cache hit" $(ls -t $E/results/server_$2_*.log | head -1) | tail -2 | tee -a $LOG
}

Q=Qwen/Qwen3.6-27B
G=google/gemma-4-31B-it
if [ "$STAGE" = all ] || [ "$STAGE" = pilot ]; then
  serve qwen $Q 16384 && run pilot qwen $Q qwen 53; stop
  serve gemma $G 8192 && run pilot gemma $G gemma4 12
  echo "[s6] PILOT DONE $(date -u +%H:%M:%S)" | tee -a $LOG
  [ "$STAGE" = pilot ] && { stop; exit 0; }
  rm -f /root/s6_go /root/s6_stop
  for i in $(seq 1 $((WAIT_MIN * 2))); do
    [ -f /root/s6_go ] || [ -f /root/s6_stop ] && break
    [ $((i % 2)) = 0 ] && echo "[s6] waiting for go/stop ($((i / 2)) min)" | tee -a $LOG
    sleep 30
  done
  if [ ! -f /root/s6_go ]; then
    echo "[s6] no go (stop file or timeout): exiting after the pilot" | tee -a $LOG
    stop; exit 0
  fi
  echo "[s6] GO received $(date -u +%H:%M:%S): $(cat /root/s6_go)" | tee -a $LOG
  curl -sf http://127.0.0.1:8000/health > /dev/null || { stop; serve gemma $G 8192; }
else
  serve gemma $G 8192
fi
run main gemma $G gemma4 12; stop
serve qwen $Q 16384 && run main qwen $Q qwen 64; stop
echo "[s6] all done" | tee -a $LOG
