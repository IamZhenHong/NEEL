#!/usr/bin/env bash
# R28 on one A100 80GB: serve each open model with vLLM's OpenAI-compatible server (native tool-call parser) and run
# run_api.py against it.  WHICH="llama mistral" or WHICH="glm" [N=10] [LIMIT=n for a smoke run]
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=vigilante/exp/r28_authority_families
R=$E/results
mkdir -p $R
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod_${WHICH// /_}.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
PY=/root/vllm/bin/python

serve() {   # name, then vllm serve args
  local name=$1; shift
  echo "[r28] serving $name: $*" | tee -a $R/pod_${name}.log
  /root/vllm/bin/vllm serve "$@" --served-model-name m --port 8000 --gpu-memory-utilization 0.92 --seed 0 \
      --enable-auto-tool-choice > $R/server_${name}.log 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:8000/health > /dev/null && { echo "[r28] $name up after $((i*10))s" | tee -a $R/pod_${name}.log; return 0; }
    kill -0 $SPID 2>/dev/null || { echo "[r28] $name server died" | tee -a $R/pod_${name}.log; tail -40 $R/server_${name}.log; return 1; }
    sleep 10
  done
  echo "[r28] $name server timeout" | tee -a $R/pod_${name}.log; return 1
}
stop() { kill $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null; pkill -9 -f "vllm serve"; pkill -9 -f EngineCore; sleep 10; nvidia-smi --query-gpu=memory.used --format=csv; }
run() {     # tag, thinking flag (0/1), extra json
  local tag=$1 th=$2 extra=${3:-'{}'}
  $PY -m vigilante.exp.r28_authority_families.run_api --backend vllm --base-url http://127.0.0.1:8000/v1 --model m \
      --tag $tag $([ "$th" = 1 ] && echo --thinking) --extra "$extra" --workers 64 --n ${N:-10} ${LIMIT:+--limit $LIMIT} \
      --out $R/$tag 2>&1 | tee -a $R/pod_$tag.log
}

for w in $WHICH; do
  case $w in
    llama)
      serve llama casperhansen/llama-3.3-70b-instruct-awq --max-model-len 24576 --tool-call-parser llama3_json \
          --chat-template vigilante/pod/templates/tool_chat_template_llama3.1_json.jinja \
        && run llama-off 0; stop ;;
    mistral)
      serve mistral mistralai/Mistral-Small-24B-Instruct-2501 --tokenizer-mode mistral --config-format mistral \
          --load-format mistral --tool-call-parser mistral --max-model-len 32768 \
        && run mistral-off 0; stop ;;
    glm)
      serve glm zai-org/GLM-4.7-Flash --tool-call-parser glm47 --reasoning-parser glm45 --max-model-len 32768 \
        && run glm-off 0 '{"chat_template_kwargs": {"enable_thinking": false}}' \
        && run glm-think 1 '{"chat_template_kwargs": {"enable_thinking": true}}'; stop ;;
  esac
done
echo "[r28] all done ($WHICH)" | tee -a $R/pod_done.log
