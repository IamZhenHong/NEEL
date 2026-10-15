#!/usr/bin/env bash
# Launch the s1 pod (s1-1) through the guarded launcher; run from anywhere, in the background:
#   nohup bash selfknow/s1_persistence/launch.sh > selfknow/s1_persistence/launch.log 2>&1 &
# The HF token is read from .env by regex (never printed). One A100 80GB. Attempt 1 (20:57) was stopped during the
# code sync ($0.36): the full-repo tar ran at ~0.6 MB/s with siblings syncing at once, so folders this run does not
# import are excluded (it needs vigilante/src + vigilante/pod + this folder + pyproject/uv.lock). Cap: what is left of $3.
cd "$(dirname "$0")/../.."
HF_TOKEN=$(python3 -c 'import re;print(re.search(r"HF_TOKEN\s*=\s*\"([^\"]+)\"", open(".env").read()).group(1))')
export HF_TOKEN
X=()
for d in cot eval_affect grievance jlens persona retention self_attribution social_feedback state_attack story_imprinting \
         tom transcripts user_autonomy vigilante/exp vigilante/results vigilante/logs selfknow/s2_role_wins \
         selfknow/s3_saydo_causes selfknow/s4_accumulation selfknow/s5_seen_unbiased selfknow/s6_reflective_identity \
         selfknow/s7_selfmodel_lag; do X+=(--exclude "$d"); done
exec caffeinate -i uv run python infra/launch_run.py --name s1-1 --allow-concurrent --max-minutes ${MAXMIN:-85} \
  --max-cost ${MAXCOST:-2.6} --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 \
  --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters "${X[@]}" \
  --results-dir selfknow/s1_persistence/results -- bash selfknow/s1_persistence/pod.sh
