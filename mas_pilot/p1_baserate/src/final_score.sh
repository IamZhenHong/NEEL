#!/usr/bin/env bash
# Final P1 scoring (run from the repo root after both pods' results are pulled). Text outputs under results/.
set -euo pipefail
B=mas_pilot/p1_baserate
OUT=$B/results/analysis; mkdir -p "$OUT"
for M in gemma qwen; do
  D=$B/results/main_$M
  [ -d "$D" ] || { echo "missing $D"; continue; }
  echo "################ $M ################"
  python3 -I $B/src/prune_scan.py "$D" --json "$D/prunes.json"
  echo "--- (i) all episodes ---"
  python3 -I $B/src/analyze_main.py "$OUT/${M}_all.json" "$M=$D"
  echo "--- (ii) excluding each trajectory's first-prune episode and later ---"
  python3 -I $B/src/analyze_main.py "$OUT/${M}_preprune.json" --exclude-after-prune "$M=$D"
  echo "--- per-episode curves, anchor, decision points ---"
  python3 -I $B/src/anchor_and_points.py "$OUT/${M}_anchor_points.json" "$M=$D"
done
echo "################ P1d judge cross-check (40 episodes, both models) ################"
python3 -I $B/src/judge_check.py "$OUT/judge_check.json" "$B/results/api_ledger.jsonl" $B/results/main_gemma $B/results/main_qwen
