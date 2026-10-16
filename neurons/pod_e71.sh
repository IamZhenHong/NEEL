#!/usr/bin/env bash
# Owner decision D1 (README): e71 despite e70's stop rule; e71 gets the same stability stop rule; if it passes, the
# authority half of e72 (thinking-on episodes; predictors run locally). e73 and the e72 false-claim half are not run.
set -uo pipefail
cd "$(dirname "$0")/.."
R=neurons/results/e71
mkdir -p $R
F="Loading weights\|Fetching\|it/s\]\|falling back\|UserWarning\|warn("
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
export CUBLAS_WORKSPACE_CONFIG=:4096:8 VLLM_USE_FLASHINFER_SAMPLER=0
bash neurons/setup_vllm.sh 2>&1 | tail -4 | tee -a $R/pod.log
/root/vllm/bin/python -m neurons.pod_gen --jobs neurons/data/gen_jobs.jsonl neurons/data/gen_jobs_e71r.jsonl \
  --exps e71 e71r --out $R/gen 2>&1 | grep --line-buffered -v "$F" | tee -a $R/pod.log
echo "[neurons] gen e71 exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
python -m neurons.pod_main --kind qwen36 --out $R --stages e71 2>&1 | grep --line-buffered -v "$F" | tee -a $R/pod.log
echo "[neurons] e71 exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
if ! python -c "import json,sys; sys.exit(1 if json.load(open('$R/STOP_RULE_e71.json'))['fires'] else 0)"; then
  echo "[neurons] e71 STOP RULE FIRES (or e71 failed): e72 authority not run" | tee -a $R/pod.log
  exit 0
fi
/root/vllm/bin/python -m neurons.pod_gen --jobs neurons/data/gen_jobs.jsonl --exps e72a --out $R/gen 2>&1 \
  | grep --line-buffered -v "$F" | tee -a $R/pod.log
echo "[neurons] gen e72a exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
python -m neurons.pod_main --kind qwen36 --out $R --stages e72a select corpus 2>&1 \
  | grep --line-buffered -v "$F" | tee -a $R/pod.log
echo "[neurons] e72a exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
du -sh $R | tee -a $R/pod.log
