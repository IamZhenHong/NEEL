#!/usr/bin/env bash
# s7 launcher wrapper: guarded launcher (AGENT_BRIEF rule 5) with the HF token read from .env by regex (never printed).
# usage: bash selfknow/s7_selfmodel_lag/launch.sh <pod-name> <pilot|main> <max-minutes> <max-cost> <wait-min> > log 2>&1
# A100s only: "NVIDIA H100 80GB HBM3" is not in infra/runpod_guard.py's allow-list (the guard raises ValueError).
# The code sync leaves out every unrelated project folder (the pilot's full-tree sync took 12 min); the pod needs only
# this folder plus pyproject.toml / uv.lock.
cd /Users/zhenhongseng/projects/ai_safety/NEEL
export HF_TOKEN=$(python3 -c "import re;print(re.search(r'HF_TOKEN\s*=\s*[\"\']([^\"\']+)', open('.env').read()).group(1))")
EXCL=()
for d in story_imprinting/data/source story_imprinting/train/adapters selfknow/s7_selfmodel_lag/adapters \
         selfknow/s7_selfmodel_lag/adapters_pod retention persona vigilante story_imprinting grievance user_autonomy \
         state_attack jlens social_feedback tom self_attribution eval_affect cot transcripts; do
  EXCL+=(--exclude "$d")
done
exec caffeinate -i uv run python infra/launch_run.py --name "$1" --allow-concurrent --max-minutes "$3" --max-cost "$4" \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 "${EXCL[@]}" \
  --results-dir selfknow/s7_selfmodel_lag/results/$2 -- env MODE=$2 WAIT_MIN=$5 bash selfknow/s7_selfmodel_lag/pod.sh
