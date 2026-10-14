#!/usr/bin/env bash
# R41 (queue item 47) on one A100 80GB: Llama 3.1 8B Instruct served by vLLM with the Persona Cartography Initiative
# adapters (amplifier, suppressor, control) at three scales; smoke, full run, then the Qwen3-14B fabrication judge.
#   uv run python infra/launch_run.py --name vig-r41 ... --results-dir vigilante/exp/r41_initiative/results -- bash vigilante/pod/r41.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
E=vigilante/exp/r41_initiative
R=$E/results
mkdir -p $R
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
PY=/root/vllm/bin/python
$PY -m vigilante.exp.r41_initiative.scale_adapters --out /root/r41_adapters 2>&1 | tee -a $R/pod.log
LORAS=$(cd /root/r41_adapters && ls -d *_* | awk '{printf "%s=/root/r41_adapters/%s ", $1, $1}')
echo "[r41] lora modules: $LORAS" | tee -a $R/pod.log
vllm serve meta-llama/Llama-3.1-8B-Instruct --served-model-name m --port 8000 --gpu-memory-utilization 0.85 --seed 0 \
    --enable-auto-tool-choice --tool-call-parser llama3_json \
    --chat-template vigilante/pod/templates/tool_chat_template_llama3.1_json.jinja --max-model-len 16384 \
    --enable-lora --max-lora-rank 64 --max-loras 9 --max-cpu-loras 12 --lora-modules $LORAS > $R/server.log 2>&1 &
SPID=$!
up=0
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/health > /dev/null && { up=1; echo "[r41] server up after $((i*10))s" | tee -a $R/pod.log; break; }
  kill -0 $SPID 2>/dev/null || break
  sleep 10
done
if [ $up = 0 ]; then echo "[r41] server failed" | tee -a $R/pod.log; tail -60 $R/server.log; exit 1; fi
$PY -m vigilante.exp.r41_initiative.run --jobs $E/jobs_llama.jsonl --out $R --limit 3 2>&1 | tee -a $R/pod.log
$PY - <<'PY' 2>&1 | tee -a $R/pod.log
import json
eps = [json.loads(l) for l in open("vigilante/exp/r41_initiative/results/episodes_agent_smoke.jsonl")]
err = sum(any(t.get("finish") == "api_error" for t in e["turns"]) for e in eps)
calls = sum(bool(e["calls"]) for e in eps)
print(f"[r41] smoke: {len(eps)} agent episodes, {calls} with tool calls, {err} with api errors", flush=True)
if err > len(eps) // 2:
    raise SystemExit(1)
PY
[ ${PIPESTATUS[0]} = 0 ] || { echo "[r41] smoke failed" | tee -a $R/pod.log; exit 1; }
$PY -m vigilante.exp.r41_initiative.run --jobs $E/jobs_llama.jsonl --out $R 2>&1 | tee -a $R/pod.log
kill $SPID; sleep 5; pkill -9 -f "vllm serve"; pkill -9 -f EngineCore; sleep 10
$PY -m vigilante.exp.r41_initiative.judge_pod --chat $R/episodes_chat.jsonl --out $R/judge_qwen.json 2>&1 | tee -a $R/pod.log
echo "[r41] pod done" | tee -a $R/pod.log
