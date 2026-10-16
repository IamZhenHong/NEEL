#!/usr/bin/env bash
# e64 pilot pod (one A100 80GB): train T8c and T9, then the gate quantities and the family base rates.
#   HF  (project venv): smoke on Qwen3-0.6B; train T8c + T9 -> results/adapters; vectors (sigma, placebo); B_hf VB-EM
#       (10 samples) + 24 timing episodes; T6 at 2/4/8 sigma VB-EM (10 samples each).
#   vLLM (own venv):    smoke on Qwen3-0.6B; B VB-EM (10) + 288 base-rate episodes; T8, T9 VB-EM (10 each).
# Judging happens locally afterwards. Launch (README "Runs"):
#   caffeinate -i uv run python infra/launch_run.py --name e64-1 --allow-concurrent --max-minutes 80 --max-cost 2.40 \
#     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 --exclude story_imprinting \
#     --exclude grievance --exclude selfknow/s7_selfmodel_lag/adapters --exclude vigilante/exp/q22_archetypes/results_r15 \
#     --sync-extra selfknow/s7_selfmodel_lag/adapters/main/s1_C_100/adapter_config.json \
#     --sync-extra selfknow/s7_selfmodel_lag/adapters/main/s1_C_100/adapter_model.safetensors \
#     --results-dir vibes/e64_layer_map/results -- bash vibes/e64_layer_map/pod_pilot.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
E=vibes/e64_layer_map; OUT=$E/results/pilot; AD=$E/results/adapters
mkdir -p $OUT $AD
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $OUT/run.log
$PY -c "import torch, transformers, peft; print('[versions] torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee -a $OUT/run.log
test -s selfknow/s7_selfmodel_lag/adapters/main/s1_C_100/adapter_model.safetensors || { echo "[e64] T8 adapter missing"; exit 1; }

# vLLM venv in the background (network + CPU) while the HF part runs
( bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1; echo $? > /root/vllm_setup.rc ) &

# ---- smoke on Qwen3-0.6B (HF stage incl. batteries + hooks + episodes, and training); stops before the paid part
$PY - <<'PY'
import torch
g = torch.Generator().manual_seed(0)
ax = {h: (lambda v: v / v.norm())(torch.randn(1024, generator=g)) for h in range(16, 29)}
torch.save({"axis": ax, "tau": {h: 0.0 for h in ax}}, "/tmp/e64_tiny_axis.pt")
PY
$PY -m vibes.e64_layer_map.stage_hf --plan tiny --model Qwen/Qwen3-0.6B --axis /tmp/e64_tiny_axis.pt --out /tmp/e64_tiny_hf \
    > /tmp/e64_tiny_hf.log 2>&1 || { tail -40 /tmp/e64_tiny_hf.log; echo "[smoke] HF FAILED"; exit 1; }
$PY -m vibes.e64_layer_map.train_adapters --which T8c T9 --model Qwen/Qwen3-0.6B --n-cap 32 --max-len 256 --micro-batch 4 \
    --out /tmp/e64_tiny_ad > /tmp/e64_tiny_ad.log 2>&1 || { tail -40 /tmp/e64_tiny_ad.log; echo "[smoke] train FAILED"; exit 1; }
test -s /tmp/e64_tiny_ad/T9/adapter_model.safetensors && test -s /tmp/e64_tiny_hf/episodes_T7.jsonl \
    && test -s /tmp/e64_tiny_hf/outputs_T6p.jsonl || { echo "[smoke] missing outputs"; exit 1; }
echo "[smoke] HF + training ok" | tee -a $OUT/run.log

# ---- 1. adapters (T8c ~50 steps, T9 218 of 435 steps)
$PY -m vibes.e64_layer_map.train_adapters --which T8c T9 --out $AD 2>&1 | tee -a $OUT/train.log
test -s $AD/T9/adapter_model.safetensors && test -s $AD/T8c/adapter_model.safetensors || { echo "[e64] training failed"; exit 1; }

# ---- 2. vLLM: smoke on Qwen3-0.6B, then the pilot (B episodes + VB-EM for B, T8, T9)
for i in $(seq 1 60); do [ -f /root/vllm_setup.rc ] && break; echo "[e64] waiting for vLLM setup"; sleep 20; done
VRC=$(cat /root/vllm_setup.rc 2>/dev/null || echo 9)
if [ "$VRC" = 0 ]; then
  /root/vllm/bin/python -m vibes.e64_layer_map.stage_vllm --plan tiny --model Qwen/Qwen3-0.6B --gpu-mem 0.5 \
      --out /tmp/e64_tiny_vllm > /tmp/e64_tiny_vllm.log 2>&1 \
    && test -s /tmp/e64_tiny_vllm/episodes_T3.jsonl && echo "[smoke] vLLM ok" | tee -a $OUT/run.log \
    && /root/vllm/bin/python -m vibes.e64_layer_map.stage_vllm --plan pilot --out $OUT 2>&1 | tee -a $OUT/vllm.log \
    || { tail -40 /tmp/e64_tiny_vllm.log; echo "[e64] vLLM part FAILED" | tee -a $OUT/run.log; }
else
  tail -30 $OUT/setup_vllm.log; echo "[e64] vLLM setup FAILED ($VRC)" | tee -a $OUT/run.log
fi

# ---- 3. HF pilot (vectors, B_hf, T6 sweep)
$PY -m vibes.e64_layer_map.stage_hf --plan pilot --out $OUT 2>&1 | tee -a $OUT/hf.log
rm -rf /root/hf/hub/models--Qwen--Qwen3-0.6B
echo "[e64] pilot pod done" | tee -a $OUT/run.log
