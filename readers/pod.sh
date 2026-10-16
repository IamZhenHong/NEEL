#!/usr/bin/env bash
# readers stage A (pilot pod, before readers-freeze): e66-e68 episodes with vLLM, then the gate (AO, J-lens, NLA)
# and the e65 pilot (8 of 80 transcripts) with HF. Launch (from repo root):
#   caffeinate -i uv run python infra/launch_run.py --name readers-1 --allow-concurrent --max-minutes 110 --max-cost 3.3 \
#     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 230 --exclude ... \
#     --results-dir readers/results/podA -- bash readers/pod.sh
# Local tests: uv run python -m readers.test_local --kind tiny36|qwen06; uv run python -m readers.pod_read gate --kind tiny36 ...
set -uo pipefail
cd "$(dirname "$0")/.."
R=readers/results/podA
mkdir -p $R
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
df -h /root | tail -1 | tee -a $R/pod.log
bash readers/setup_vllm.sh 2>&1 | tail -4 | tee -a $R/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m readers.pod_gen --jobs readers/data/gen_jobs.jsonl --out $R/gen 2>&1 \
  | grep --line-buffered -v "Loading safetensors\|Fetching\|it/s\]" | tee -a $R/pod.log
echo "[readers] gen exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
python -m readers.pod_read gate --out $R/gate --nla-max-new 200 2>&1 \
  | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a $R/pod.log
echo "[readers] gate exit ${PIPESTATUS[0]}" | tee -a $R/pod.log
du -sh $R | tee -a $R/pod.log
