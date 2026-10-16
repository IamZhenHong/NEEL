#!/usr/bin/env bash
# Guarded launch (AGENT_BRIEF rule 5), from the repo root, in the background:
#   bash vibes/e63_register/launch.sh <name> <max-minutes> <max-cost> [pod script] > vibes/e63_register/results/launch_<name>.log 2>&1
# (pod script: vibes/e63_register/pod.sh = pilot + hold, the first launch; pod_main.sh = the main run after e63-freeze)
# The HF token is read from .env by a regex and only placed in the environment, never printed. The sync leaves out
# every folder e63 does not use (the pod needs only vibes/, pyproject and uv.lock; setup_vllm.sh is copied in here).
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME=$1; MIN=$2; COST=$3; SCRIPT=${4:-vibes/e63_register/pod.sh}; RDIR=${5:-vibes/e63_register/results/pod}
export HF_TOKEN="$(python3 -c "import re;m=re.search(r'HF_TOKEN\s*=\s*[\"\x27]?([^\"\x27\s]+)',open('.env').read());print(m.group(1) if m else '')")"
[ -n "$HF_TOKEN" ] || { echo "no HF_TOKEN in .env"; exit 1; }
df -h . | tail -1
EX=()
for p in story_imprinting grievance selfknow vigilante persona retention state_attack jlens user_autonomy \
         social_feedback cot tom self_attribution eval_affect readers transcripts prep \
         vibes/e61_free_vars vibes/e62_pull vibes/e64_layer_map vibes/e63_register/results; do
  EX+=(--exclude "$p")
done
exec caffeinate -i uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MIN" \
    --max-cost "$COST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 "${EX[@]}" \
    --results-dir "$RDIR" -- bash "$SCRIPT"
