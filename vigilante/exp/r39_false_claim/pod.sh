#!/usr/bin/env bash
# R39 on one A100 80GB: Gemma 4 12B reads (primary), then Qwen3.6-27B reads (secondary). transformers, project venv.
set -uo pipefail
cd "$(dirname "$0")/../../.."
R=vigilante/exp/r39_false_claim/results
mkdir -p $R
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/pod.log
tar -xzf vigilante/exp/r39_false_claim/data/cancel_wording_cells.tgz -C vigilante/exp/r39_false_claim/data
python -m vigilante.exp.r39_false_claim.pod_r39 --stage gemma --out $R 2>&1 | tee -a $R/pod.log || echo "[r39] gemma FAILED" | tee -a $R/pod.log
python -m vigilante.exp.r39_false_claim.pod_r39 --stage qwen --out $R 2>&1 | tee -a $R/pod.log || echo "[r39] qwen FAILED" | tee -a $R/pod.log
echo "[r39] all done" | tee -a $R/pod.log
