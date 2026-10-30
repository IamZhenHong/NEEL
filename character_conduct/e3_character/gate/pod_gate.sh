#!/usr/bin/env bash
# CC-E3 step 0 on one A100 80GB (DESIGN.md §Step 0): for each original checkpoint, serve it with vLLM's
# OpenAI-compatible server (native tool-call parser, served as "m"), run AgentHarm harmful + benign (test_public)
# through inspect_evals with --no-score (generation only; judging is local), then the blackmail jobs.
#   MODELS="qwen llama" EPOCHS=5 [LIMIT=n for a smoke run] bash character_conduct/e3_character/gate/pod_gate.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
E=character_conduct/e3_character
R=${OUT:-$E/results/gate}
mkdir -p "$R"
EPOCHS=${EPOCHS:-5}
MODELS=${MODELS:-"qwen llama"}
log() { echo "[e3-gate] $*" | tee -a "$R/pod.log"; }

bash $E/pod/setup_vllm.sh 2>&1 | tee -a "$R/pod.log" || { log "vllm setup failed"; exit 1; }
export PATH=/root/vllm/bin:$HOME/.local/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
if [ ! -x /root/insp/bin/inspect ]; then
  uv venv -q /root/insp --python 3.12 && VIRTUAL_ENV=/root/insp uv pip install -q inspect-ai==0.3.285 inspect-evals==0.26.0 openai \
    || { log "inspect venv failed"; exit 1; }
fi
/root/insp/bin/python -c "import inspect_ai, inspect_evals; print('[e3-gate] inspect_ai', inspect_ai.__version__)" | tee -a "$R/pod.log"
VIRTUAL_ENV=/root/insp uv pip freeze > "$R/insp_freeze.txt" 2>/dev/null || true

serve() {   # name, then vllm serve args
  local name=$1; shift
  log "serving $name: $*"
  /root/vllm/bin/vllm serve "$@" --served-model-name m --port 8000 --gpu-memory-utilization 0.90 --seed 0 \
      --max-model-len 16384 --enable-auto-tool-choice > "$R/server_${name}.log" 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do
    curl -sf http://127.0.0.1:8000/health > /dev/null && { log "$name up after $((i*10))s"; return 0; }
    kill -0 $SPID 2>/dev/null || { log "$name server died"; tail -40 "$R/server_${name}.log"; return 1; }
    sleep 10
  done
  log "$name server timeout"; return 1
}
stop() { kill $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null; pkill -9 -f "vllm serve"; pkill -9 -f EngineCore; sleep 10
         nvidia-smi --query-gpu=memory.used --format=csv | tee -a "$R/pod.log"; }

agentharm() {   # name
  local name=$1 t
  for t in agentharm agentharm_benign; do
    log "$name $t: epochs $EPOCHS ${LIMIT:+limit $LIMIT}"
    LOCAL_BASE_URL=http://127.0.0.1:8000/v1 LOCAL_API_KEY=x /root/insp/bin/inspect eval "inspect_evals/$t" \
      --model openai-api/local/m -T split=test_public \
      -T refusal_judge=openai-api/local/m -T semantic_judge=openai-api/local/m \
      --epochs "$EPOCHS" --temperature 0.7 --top-p 0.9 --max-connections 64 --no-score \
      --log-dir "$R/inspect/$name" --log-format eval --display plain ${LIMIT:+--limit $LIMIT} \
      2>&1 | tee -a "$R/inspect_${name}_${t}.log"
  done
}

blackmail() {   # name
  /root/vllm/bin/python -m character_conduct.e3_character.gate.run_bm --jobs $E/gate/materials/bm_jobs.jsonl \
    --out "$R/bm_$1.jsonl" --model-tag "$1" ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a "$R/pod.log"
}

for m in $MODELS; do
  case $m in
    qwen)
      serve qwen Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --tool-call-parser hermes \
        && agentharm qwen && blackmail qwen; stop ;;
    llama)
      serve llama meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
          --tool-call-parser llama3_json --chat-template $E/pod/templates/tool_chat_template_llama3.1_json.jinja \
        && agentharm llama && blackmail llama; stop ;;
  esac
done
ls -la "$R" "$R"/inspect/* 2>/dev/null | tee -a "$R/pod.log"
log "all done ($MODELS)"
