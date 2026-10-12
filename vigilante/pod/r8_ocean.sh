#!/usr/bin/env bash
# R8: Qwen3-32B bf16 + Persona Cartography OCEAN adapters (one adapter resident at a time), A100 80GB.
#   uv run python infra/launch_run.py --name vig-r8 --max-minutes 80 --max-cost 2.4 ... \
#       --results-dir vigilante/exp/r8_ocean/results -- bash vigilante/pod/r8_ocean.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=vigilante/exp/r8_ocean/results; mkdir -p $OUT
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $OUT/run.log
/root/vllm/bin/python - <<'PY' 2>&1 | tee -a $OUT/run.log
import json
from huggingface_hub import snapshot_download
ad = json.load(open("vigilante/exp/r8_ocean/adapters.json"))
snapshot_download("persona-cartography/monorepo", repo_type="dataset", local_dir="/root/ocean",
                  allow_patterns=[f"{p}/adapter_*" for p in ad.values()])
print("[r8] adapters downloaded:", sorted(ad), flush=True)
PY
LORAS=$(/root/vllm/bin/python -c "import json;ad=json.load(open('vigilante/exp/r8_ocean/adapters.json'));print(' '.join(f'--lora {k}=/root/ocean/{v}' for k,v in ad.items()))")
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -m vigilante.src.pod_run --jobs vigilante/exp/r8_ocean/jobs.jsonl --out $OUT --model Qwen/Qwen3-32B \
    --fmt qwen3 --max-model-len 6144 $LORAS ${LIMIT:+--limit $LIMIT} 2>&1 | tee -a $OUT/run.log
echo "[r8] pod done" | tee -a $OUT/run.log
