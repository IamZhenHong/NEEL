#!/usr/bin/env bash
# s4 on one A100 80GB: Qwen3.6-27B, then Gemma 4 31B, as separate vLLM processes (thinking off).
#   STAGE=pilot  -> jobs_pilot.jsonl -> results/pilot/{qwen,gemma}
#   STAGE=main   -> jobs_main.jsonl  -> results/main/{qwen,gemma}
# STAGE=main runs Gemma first (confirmatory), then Qwen (exploratory: failed the gate; README Amendment).
# Both model downloads start in the background at once (project venv), while the vLLM venv installs.
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${STAGE:?set STAGE=pilot|main}"
E=selfknow/s4_accumulation
R=$E/results/$STAGE
mkdir -p $R
export HF_HOME=/root/hf
for M in Qwen/Qwen3.6-27B google/gemma-4-31B-it; do
  /root/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('$M', allow_patterns=['*.json','*.safetensors','*.jinja','tokenizer*','*.model','*.txt']); print('[s4] cached $M', flush=True)" >> $R/download.log 2>&1 &
done
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
PY=/root/vllm/bin/python
run_model () {   # $1 = qwen|gemma
  if [ "$1" = qwen ]; then M=Qwen/Qwen3.6-27B; F=qwen; MML=32768; U=0.92; else M=google/gemma-4-31B-it; F=gemma4; MML=12288; U=0.93; fi
  $PY -m selfknow.s4_accumulation.pod_run --jobs $E/jobs_$STAGE.jsonl --out $R/$1 --model $M --fmt $F \
      --max-model-len $MML --gpu-util $U ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a $R/pod.log || echo "[s4] $1 FAILED" | tee -a $R/pod.log
  pkill -9 -f EngineCore; sleep 5
}
# pilot: qwen then gemma (as run on 2026-10-15). main: gemma (confirmatory) first, then qwen (exploratory, gate failed)
if [ "$STAGE" = main ]; then ORDER="gemma qwen"; else ORDER="qwen gemma"; fi
for m in $ORDER; do run_model $m; cat $R/download.log >> $R/pod.log; done
echo "[s4] pod done ($STAGE)" | tee -a $R/pod.log
