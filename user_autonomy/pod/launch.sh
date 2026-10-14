#!/usr/bin/env bash
# launch.sh NAME MINUTES COST RESULTS_DIR [extra launcher args...] -- POD_CMD...
# Reads HF_TOKEN from NEEL/.env with a Python regex (never printed; .env uses "KEY = value" so `source` fails).
cd "$(dirname "$0")/../.."
export HF_TOKEN=$(python3 -c "import re;t=open('.env').read();m=re.search(r'^\s*(?:export\s+)?HF_TOKEN\s*=\s*(.+)$',t,re.M);print(m.group(1).strip().strip('\"\'') if m else '')")
NAME=$1; MIN=$2; COST=$3; RES=$4; shift 4
exec caffeinate -i uv run python infra/launch_run.py --name "$NAME" --max-minutes "$MIN" --max-cost "$COST" \
    --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --allow-concurrent \
    --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
    --results-dir "$RES" "$@"
