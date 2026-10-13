#!/usr/bin/env bash
# R32c (+ R30d on Qwen). WHICH=offline: Qwen3.6-27B (R30d + R32c) then Gemma 4 31B (R32c), offline engine.
# WHICH=server: Llama 3.3 70B AWQ then Mistral Small 3 24B behind vLLM's OpenAI server (as R28), run_server.py.
#   uv run python infra/launch_run.py --name vig-r32c-$WHICH ... --results-dir vigilante/exp/r32_sales/results/c_$WHICH \
#       -- env WHICH=$WHICH bash vigilante/exp/r32_sales/pod_c.sh      [LIMIT=n for a smoke run]
set -uo pipefail
cd "$(dirname "$0")/../../.."
: "${WHICH:?offline or server}"
E=vigilante/exp/r32_sales
R=$E/results/c_$WHICH
mkdir -p $R
if [ "$WHICH" = offline ]; then
  mkdir -p $R/qwen $R/gemma
  cat vigilante/exp/r30_social_pressure/jobs_d_qwen.jsonl $E/jobs_c_qwen.jsonl > $R/qwen/jobs.jsonl
  MODEL=Qwen/Qwen3.6-27B FMT=qwen MML=40960 JOBS=$R/qwen/jobs.jsonl OUT=$R/qwen bash vigilante/pod/run_jobs.sh || echo "[r32c] qwen FAILED" | tee -a $R/pod.log
  pkill -9 -f EngineCore; sleep 10
  MODEL=google/gemma-4-31B-it FMT=gemma4 MML=8192 JOBS=$E/jobs_c_gemma.jsonl OUT=$R/gemma bash vigilante/pod/run_jobs.sh || echo "[r32c] gemma FAILED" | tee -a $R/pod.log
else
  bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
  export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
  serve() {   # name, then vllm serve args (as R28's pod_server.sh)
    local name=$1; shift
    echo "[r32c] serving $name: $*" | tee -a $R/pod.log
    /root/vllm/bin/vllm serve "$@" --served-model-name m --port 8000 --gpu-memory-utilization 0.92 --seed 0 \
        --enable-auto-tool-choice > $R/server_${name}.log 2>&1 &
    SPID=$!
    for i in $(seq 1 180); do
      curl -sf http://127.0.0.1:8000/health > /dev/null && { echo "[r32c] $name up after $((i*10))s" | tee -a $R/pod.log; return 0; }
      kill -0 $SPID 2>/dev/null || { echo "[r32c] $name server died" | tee -a $R/pod.log; tail -40 $R/server_${name}.log; return 1; }
      sleep 10
    done
    echo "[r32c] $name server timeout" | tee -a $R/pod.log; return 1
  }
  stop() { kill $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null; pkill -9 -f "vllm serve"; pkill -9 -f EngineCore; sleep 10; }
  run() { /root/vllm/bin/python -m vigilante.exp.r32_sales.run_server --jobs $E/jobs_c_$1.jsonl --out $R/$1 ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a $R/pod.log; }
  serve llama casperhansen/llama-3.3-70b-instruct-awq --max-model-len 24576 --tool-call-parser llama3_json \
      --chat-template vigilante/pod/templates/tool_chat_template_llama3.1_json.jinja && run llama; stop
  serve mistral mistralai/Mistral-Small-24B-Instruct-2501 --tokenizer-mode mistral --config-format mistral \
      --load-format mistral --tool-call-parser mistral --max-model-len 32768 && run mistral; stop
fi
echo "[r32c] all done ($WHICH)" | tee -a $R/pod.log
