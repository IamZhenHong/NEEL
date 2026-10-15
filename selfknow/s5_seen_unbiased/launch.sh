#!/usr/bin/env bash
# Guarded launch (AGENT_BRIEF rule 5). Usage, from the repo root, in the background:
#   bash selfknow/s5_seen_unbiased/launch.sh <name> <max-minutes> <max-cost> <results-subdir> <pod script>
# e.g. bash selfknow/s5_seen_unbiased/launch.sh s5-1 45 1.4 pilot selfknow/s5_seen_unbiased/pod_pilot.sh
# The HF token is read from .env by a regex and only placed in the environment, never printed.
# GPU list: A100 80GB types only (infra/runpod_guard.py has no H100 entry in its allow-list).
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME=$1; MIN=$2; COST=$3; SUB=$4; SCRIPT=$5
export HF_TOKEN="$(python3 -c "import re;m=re.search(r'HF_TOKEN\s*=\s*[\"\x27]?([^\"\x27\s]+)',open('.env').read());print(m.group(1) if m else '')")"
[ -n "$HF_TOKEN" ] || { echo "no HF_TOKEN in .env"; exit 1; }
df -h . | tail -1
exec caffeinate -i uv run python infra/launch_run.py --name "$NAME" --allow-concurrent --max-minutes "$MIN" \
    --max-cost "$COST" --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
    --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
    --results-dir "selfknow/s5_seen_unbiased/results${SUB:+/$SUB}" -- bash "$SCRIPT"
