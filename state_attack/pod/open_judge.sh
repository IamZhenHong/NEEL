#!/usr/bin/env bash
# Open-weights cross-family judge (Gemma 4 31B). Pre-downloads the model, waits for the items file (copied in later
# with a .ready marker), labels by letter probabilities, exits (the launcher pulls results/open_judge).
cd /root/NEEL
mkdir -p state_attack/results/open_judge
python -c "from huggingface_hub import snapshot_download; snapshot_download('google/gemma-4-31B-it'); print('[open-judge] model cached', flush=True)"
until [ -f state_attack/data/open_judge_items.ready ]; do echo "[open-judge] waiting for items $(date +%T)"; sleep 30; done
python -u -m state_attack.src.judge_open --run 2>&1 | tee state_attack/results/open_judge/run.log
