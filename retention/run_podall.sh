#!/usr/bin/env bash
# Retention block, 2026-11-04: Pod A (friend gate + affect), soft exits, support x mood on ONE pod.
# Smoke passes first (tiny model), one model download, then the three full runs: sequential on a 24-48 GB card,
# concurrent on >= 70 GB. Every run is resumable per cell, so a rerun after a time cap only does what is missing.
# Run from /root/NEEL inside the uv env:  uv run bash retention/run_podall.sh
set -uo pipefail
R=retention/results
mkdir -p "$R/podall_logs"
echo "[podall] smoke passes"
python retention/run_softexit.py --smoke --out /root/smoke/softexit || { echo "[podall] softexit smoke FAILED"; exit 2; }
python retention/run_userstate2x2.py --smoke --out /root/smoke/u2x2 || { echo "[podall] 2x2 smoke FAILED"; exit 2; }
echo "[podall] downloading Qwen2.5-7B-Instruct once"
python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen2.5-7B-Instruct')" || exit 3
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "[podall] GPU memory ${MEM} MiB"
if [ "${MEM:-0}" -ge 70000 ]; then
  echo "[podall] concurrent mode"
  python retention/run_podA.py --out $R/retention_podA > $R/podall_logs/podA.log 2>&1 & pa=$!
  python retention/run_softexit.py --out $R/retention_softexit > $R/podall_logs/softexit.log 2>&1 & ps=$!
  python retention/run_userstate2x2.py --out $R/retention_userstate2x2 > $R/podall_logs/u2x2.log 2>&1 & pu=$!
  while kill -0 $pa 2>/dev/null || kill -0 $ps 2>/dev/null || kill -0 $pu 2>/dev/null; do
    sleep 60
    for f in podA softexit u2x2; do echo "[$f] $(grep -E '^\[(cell|stage1|done|skip)' $R/podall_logs/$f.log | tail -1 | cut -c1-110)"; done
  done
  rc=0; wait $pa || rc=1; wait $ps || rc=1; wait $pu || rc=1
  tail -3 $R/podall_logs/*.log
  exit $rc
else
  echo "[podall] sequential mode"
  rc=0
  python retention/run_podA.py --out $R/retention_podA || rc=1
  python retention/run_softexit.py --out $R/retention_softexit || rc=1
  python retention/run_userstate2x2.py --out $R/retention_userstate2x2 || rc=1
  exit $rc
fi
