#!/usr/bin/env bash
# e64 main pod (one A100 80GB): every condition on both surfaces.
#   HF   (project venv): B_hf, T7, T6, T6p: VB-1 full + the main HF episodes (k from data/hook_config.json)
#   vLLM (own venv, installed in the background meanwhile): B, T0-T5, T8, T8c, T9: VB-1 full + the main episodes
# Judging happens locally afterwards. Launch command: README "Runs".
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
E=vibes/e64_layer_map; OUT=$E/results/main
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $OUT/run.log
for f in selfknow/s7_selfmodel_lag/adapters/main/s1_C_100/adapter_model.safetensors $E/results/adapters/T8c/adapter_model.safetensors \
         $E/results/adapters/T9/adapter_model.safetensors $E/data/vectors.pt $E/data/hook_config.json $E/jobs/main_vllm.jsonl \
         $E/jobs/main_hf.jsonl; do
  test -s $f || { echo "[e64] missing $f"; exit 1; }
done
K=$($PY -c "import json; print(json.load(open('$E/data/hook_config.json'))['k'])")
echo "[e64] T6/T6p k = $K" | tee -a $OUT/run.log

( bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1; echo $? > /root/vllm_setup.rc ) &

$PY -m vibes.e64_layer_map.stage_hf --plan main --out $OUT --vectors $E/data/vectors.pt --k $K \
    --conds B_hf T7 T6 T6p 2>&1 | tee -a $OUT/hf.log

for i in $(seq 1 60); do [ -f /root/vllm_setup.rc ] && break; echo "[e64] waiting for vLLM setup"; sleep 20; done
if [ "$(cat /root/vllm_setup.rc 2>/dev/null || echo 9)" = 0 ]; then
  /root/vllm/bin/python -m vibes.e64_layer_map.stage_vllm --plan main --out $OUT 2>&1 | tee -a $OUT/vllm.log
else
  tail -30 $OUT/setup_vllm.log; echo "[e64] vLLM setup FAILED" | tee -a $OUT/run.log
fi
echo "[e64] main pod done" | tee -a $OUT/run.log
