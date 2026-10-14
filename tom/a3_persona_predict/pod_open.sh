#!/usr/bin/env bash
# A3b on one A100 80GB: Qwen3.6-27B then Gemma 4 31B as predictors (thinking off and on), offline vLLM.
#   uv run python -m tom.a3_persona_predict.open_pred --bundle   # first, locally
#   uv run python infra/launch_run.py --name tom-a3b --max-minutes M --max-cost C --gpus "NVIDIA A100-SXM4-80GB" \
#       "NVIDIA A100 80GB PCIe" --container-gb 160 --exclude story_imprinting/data/source \
#       --exclude story_imprinting/train/adapters --results-dir tom/a3_persona_predict/results/open \
#       -- bash tom/a3_persona_predict/pod_open.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
R=tom/a3_persona_predict/results/open
mkdir -p $R
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -c "from huggingface_hub import snapshot_download as s; [s(m) for m in ('Qwen/Qwen3.6-27B', 'google/gemma-4-31B-it')]; print('[a3b] models cached', flush=True)" 2>&1 | tee -a $R/pod.log
B=tom/a3_persona_predict/data/open_bundle
until [ -f $B/READY ]; do echo "[a3b] waiting for the bundle $(date +%T)"; sleep 30; done
/root/vllm/bin/python -m tom.a3_persona_predict.open_pred --model Qwen/Qwen3.6-27B --short q36 --fmt qwen --mml 8192 \
    --out $R 2>&1 | tee -a $R/pod.log || echo "[a3b] qwen FAILED" | tee -a $R/pod.log
/root/vllm/bin/python -m tom.a3_persona_predict.open_pred --model google/gemma-4-31B-it --short g31 --fmt gemma4 \
    --mml 8192 --out $R 2>&1 | tee -a $R/pod.log || echo "[a3b] gemma FAILED" | tee -a $R/pod.log
echo "[a3b] all done" | tee -a $R/pod.log
