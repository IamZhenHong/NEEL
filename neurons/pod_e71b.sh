#!/usr/bin/env bash
# D1 pod 4: the E2 e72-authority top-up (208 peer samples) and, if both classes reach 8, e72 authority attribution,
# neuron selection and the label-corpus pass for the new neurons. (E1 + e71 ablations: pod_abl71.sh, after labels.)
set -uo pipefail
cd "$(dirname "$0")/.."
R=neurons/results/e71/pod4
mkdir -p $R/gen
cp neurons/data/e71_gen/gen_e72a.jsonl.gz $R/gen/
cp neurons/data/e71_gen/e71_summary.json $R/
F="Loading weights\|Fetching\|it/s\]\|falling back\|UserWarning\|warn("
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
export CUBLAS_WORKSPACE_CONFIG=:4096:8 VLLM_USE_FLASHINFER_SAMPLER=0
bash neurons/setup_vllm.sh 2>&1 | tail -4 | tee -a $R/pod.log
/root/vllm/bin/python -m neurons.pod_gen --jobs neurons/data/gen_jobs_e72at.jsonl --exps e72at --out $R/gen 2>&1 \
  | grep --line-buffered -v "$F" | tee -a $R/pod.log
echo "[neurons] gen e72at exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
python -m neurons.pod_main --kind qwen36 --out $R --stages e72a 2>&1 | grep --line-buffered -v "$F" | tee -a $R/pod.log
if [ -f $R/e72a_summary.json ]; then
  python -m neurons.pod_main --kind qwen36 --out $R --stages select corpus 2>&1 | grep --line-buffered -v "$F" | tee -a $R/pod.log
else
  echo "[neurons] e72 authority dropped (a class < 8 after 400 samples)" | tee -a $R/pod.log
fi
rm -f $R/gen/gen_e72a.jsonl.gz
du -sh $R | tee -a $R/pod.log
