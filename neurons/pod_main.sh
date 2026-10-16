#!/usr/bin/env bash
# neurons main pod (after neurons-freeze). Order (DESIGN "Order, gate and budget"):
#   1. gate + e70 on Qwen3.6-27B (HF); the stability stop rule is checked here and the pod stops if it fires;
#   2. vLLM: e71 greedy candidates, e72 thinking-on episodes;
#   3. HF Qwen3.6-27B: e71, e72 attribution, neuron selection, label-corpus pass, e70 ablations, specificity;
#   4. HF Qwen3-14B: e73 attribution, neuron selection, label-corpus pass.
set -uo pipefail
cd "$(dirname "$0")/.."
R=neurons/results/main
mkdir -p $R
F="Loading weights\|Fetching\|it/s\]\|falling back"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m neurons.pod_attr --kind qwen36 --out $R --stages gate e70 2>&1 | grep --line-buffered -v "$F" | tee -a $R/pod.log
echo "[neurons] e70 exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
# continue only if STOP_RULE.json exists and says the rule does not fire (a crash before it also stops the pod)
if ! python -c "import json,sys; sys.exit(1 if json.load(open('$R/STOP_RULE.json'))['fires'] else 0)"; then
  echo "[neurons] STOP RULE FIRES (or e70 failed): e71-e73 not run" | tee -a $R/pod.log
  exit 0
fi
bash neurons/setup_vllm.sh 2>&1 | tail -4 | tee -a $R/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m neurons.pod_gen --jobs neurons/data/gen_jobs.jsonl --out $R/gen 2>&1 \
  | grep --line-buffered -v "$F" | tee -a $R/pod.log
echo "[neurons] gen exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
python -m neurons.pod_main --kind qwen36 --out $R --stages e71 e72 select corpus abl70 spec70 2>&1 \
  | grep --line-buffered -v "$F" | tee -a $R/pod.log
echo "[neurons] qwen36 exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
python -m neurons.pod_main --kind qwen14 --out $R --stages e73 select corpus 2>&1 \
  | grep --line-buffered -v "$F" | tee -a $R/pod.log
echo "[neurons] qwen14 exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
du -sh $R | tee -a $R/pod.log
