#!/usr/bin/env bash
# e61 launcher wrapper: guarded launcher (AGENT_BRIEF rule 5), HF token read from .env by regex (never printed), A100s
# only, every unrelated project folder left out of the code sync (the pod needs only vibes/, infra/, pyproject, uv.lock).
# usage: bash vibes/e61_free_vars/launch.sh <pod-name> <pilot|main> <max-minutes> <max-cost> <wait-min> > log 2>&1
cd /Users/zhenhongseng/projects/ai_safety/NEEL
export HF_TOKEN=$(python3 -c "import re;print(re.search(r'HF_TOKEN\s*=\s*[\"\']([^\"\']+)', open('.env').read()).group(1))")
EXCL=()
for d in story_imprinting grievance selfknow vigilante/exp/q22_archetypes/results_r15 retention persona vigilante \
         user_autonomy state_attack jlens social_feedback tom self_attribution eval_affect cot transcripts readers \
         vibes/e61_free_vars/adapters vibes/e61_free_vars/adapters_pod vibes/batteries/results; do
  EXCL+=(--exclude "$d")
done
POD=vibes/e61_free_vars/pod.sh; [ "$2" = main ] && POD=vibes/e61_free_vars/pod_main.sh
exec caffeinate -i uv run python infra/launch_run.py --name "$1" --allow-concurrent --max-minutes "$3" --max-cost "$4" \
  --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 "${EXCL[@]}" \
  --results-dir vibes/e61_free_vars/results/$2 -- env MODE=$2 WAIT_MIN=$5 bash $POD
