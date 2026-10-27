#!/usr/bin/env bash
# EL-C4 pod (DESIGN.md §3-5; 2 x A100 80GB). Launch with launch_c4.sh, never by hand. EL-C2's pod_wood.sh recipe:
#   1. a venv pinned to Hua et al.'s versions (/root/hua: python 3.11, torch 2.7.0 cu126, transformers 4.53.2,
#      peft 0.16.0, accelerate, numpy, hf_transfer)
#   2. the pinned Nemotron Super 49B v1 and the Wood Labs LoRA, downloaded with hf_transfer; free disk logged first
#   3. pod_c4.py (its own deadline: no stage starts within 8 minutes of T0 + MAXMIN)
#   4. the model and xet caches are deleted before the pull
# env: T0, MAXMIN, OUT (evalaware_lens/c4_mediation/results/main)
set -uo pipefail
cd "$(dirname "$0")/../.."
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
mkdir -p "$OUT"
LOG=$OUT/pod.log
export PATH=$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
[ -f evalaware_lens/c2_steer/results/wood/gap.jsonl ] && echo "[el-c4] C2 row-0 replies present" | tee -a "$LOG" \
  || echo "[el-c4] C2 gap.jsonl missing: the descriptive reply reads will be skipped" | tee -a "$LOG"
if [ ! -x /root/hua/bin/python ]; then
  uv venv -q -p 3.11 /root/hua 2>&1 | tail -2 | tee -a "$LOG"
  uv pip install -q --python /root/hua/bin/python --index-url https://download.pytorch.org/whl/cu126 "torch==2.7.0" 2>&1 | tail -2 | tee -a "$LOG"
  uv pip install -q --python /root/hua/bin/python "transformers==4.53.2" "peft==0.16.0" accelerate numpy safetensors \
    sentencepiece hf_transfer 2>&1 | tail -2 | tee -a "$LOG"
fi
PY=/root/hua/bin/python
"$PY" -c "import torch, transformers, peft; print('[el-c4] torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee -a "$LOG"
df -h /root 2>/dev/null | tail -1 | sed "s/^/[el-c4] disk before the download: /" | tee -a "$LOG"
"$PY" - <<'PYEOF' 2>&1 | tail -3 | tee -a "$LOG"
from huggingface_hub import snapshot_download
from evalaware_lens.c2_steer.pod_wood import BASE, BASE_REV, LORA, LORA_REV
print(snapshot_download(BASE, revision=BASE_REV, max_workers=16))
print(snapshot_download(LORA, revision=LORA_REV))
PYEOF
"$PY" -m evalaware_lens.c4_mediation.pod_c4 --out "$OUT" 2>&1 | grep --line-buffered -v "Loading checkpoint\|it/s\]" | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[el-c4] exit $rc" | tee -a "$LOG"
rm -rf /root/hf/hub/models--* /root/hf/xet 2>/dev/null
exit "$rc"
