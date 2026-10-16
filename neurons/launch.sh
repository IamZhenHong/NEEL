#!/usr/bin/env bash
# Guarded launch for neurons pods (AGENT_BRIEF rule 5). Usage: bash neurons/launch.sh <name> <max-min> <max-cost> <pod script>
# HF token read from .env with a regex (never printed).
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_TOKEN=$(python3 -c "import re;print(re.search(r'HF_TOKEN\s*=\s*\"?([^\"\s]+)',open('.env').read()).group(1))")
exec caffeinate -i uv run python infra/launch_run.py --name "$1" --allow-concurrent --max-minutes "$2" --max-cost "$3" \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
  --exclude story_imprinting --exclude grievance --exclude selfknow --exclude persona --exclude retention \
  --exclude user_autonomy --exclude state_attack --exclude social_feedback --exclude jlens/results \
  --exclude vigilante/exp/q22_archetypes --exclude vigilante/exp/r23_long_mech --exclude vigilante/exp/q0_quick \
  --exclude vigilante/exp/r28_authority_families/results --exclude vigilante/exp/r39_false_claim/data \
  --exclude vigilante/logs --exclude readers/results --exclude vibes/e63_register/results \
  --exclude vibes/e64_layer_map/results --exclude vibes/e61_free_vars/results --exclude vibes/e62_pull/results \
  --exclude neurons/results --exclude cot --exclude tom --exclude self_attribution --exclude eval_affect \
  --results-dir "neurons/results/$5" -- bash "$4"
