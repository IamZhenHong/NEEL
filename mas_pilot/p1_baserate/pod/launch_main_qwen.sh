#!/usr/bin/env bash
# Qwen main pod launch (run from repo root after the Gemma pod has finished). Args: MAX_COST MAX_MIN
set -euo pipefail
MAXCOST="$1"; MAXMIN="$2"
export HF_TOKEN="$(python3 -I -c "
import re
for line in open('.env'):
    m = re.match(r'\s*HF_TOKEN\s*=\s*(.*)\s*\$', line)
    if m:
        print(m.group(1).strip().strip('\"').strip(\"'\"), end='')
        break
")"
[ -n "$HF_TOKEN" ] || { echo "no HF_TOKEN"; exit 1; }
caffeinate -i uv run python infra/launch_run.py \
  --max-minutes "$MAXMIN" --max-cost "$MAXCOST" --min-free-gb 1.0 \
  --gpus "NVIDIA A100 80GB PCIe" "NVIDIA A100-SXM4-80GB" \
  --container-gb 130 --bootstrap-minutes 20 \
  --name p1-ec-main-qwen --allow-concurrent --workdir "" --results-dir mas_pilot/p1_baserate/results \
  --exclude selfknow --exclude story_imprinting --exclude grievance --exclude vigilante \
  --exclude retention --exclude persona --exclude readers --exclude neurons --exclude vibes \
  --exclude tom --exclude user_autonomy --exclude state_attack --exclude eval_affect \
  --exclude social_feedback --exclude self_attribution --exclude coherence --exclude cot \
  --exclude jlens --exclude transcripts --exclude mas_pilot/p0_read --exclude mas_pilot/p1_baserate/ext/pub_traces \
  -- env MAXLEN=81920 THINK=1 GPU_UTIL=0.94 KV_OFFLOAD_GB=auto HEARTBEAT_S=300 'MM_LIMIT={"image": 0, "video": 0}' \
     bash mas_pilot/p1_baserate/pod/run_arms.sh \
     Qwen/Qwen3.6-27B qwen3.6-27b mas_pilot/p1_baserate/pod/jobs_main.tsv \
     mas_pilot/p1_baserate/results/main_qwen 14848 5 qwen3 hermes
