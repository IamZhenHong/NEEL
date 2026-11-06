#!/usr/bin/env bash
# Pod 2 of prereg retention/prereg/retention_mechanisms_v1.md: X5 (Qwen3-8B via vLLM), then X4 (7B HF steering), then X6 (J-lens, HF).
#   uv run bash retention/mech_pod2.sh
set -euo pipefail
OUT=retention/results/retention_mechanisms/pod2
mkdir -p "$OUT"
SPID=""
stop() {
  [ -n "$SPID" ] && { kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }
  pkill -f "vllm serve" 2>/dev/null || true; SPID=""
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${used:-0}" -lt 3000 ] && return 0; sleep 5
  done
  echo "[pod2] GPU still holds ${used} MiB"; return 1
}
trap 'stop || true' EXIT
echo "[pod2] $(date +%T) serving Qwen3-8B"
bash retention/vllm_serve.sh Qwen/Qwen3-8B 8000 0.90 16384 > /root/vllm_qwen3_8b.log 2>&1 &
SPID=$!
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8000/health >/dev/null && break
  kill -0 "$SPID" 2>/dev/null || { echo "[pod2] vllm exited"; tail -60 /root/vllm_qwen3_8b.log; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:8000/health >/dev/null || { echo "[pod2] vllm did not come up"; tail -60 /root/vllm_qwen3_8b.log; exit 1; }
python retention/mech_pod2.py --task x5 --url http://127.0.0.1:8000 --out "$OUT"
cp /root/vllm_qwen3_8b.log "$OUT/" || true
stop
echo "[pod2] $(date +%T) x5 done"
python retention/mech_pod2.py --task x4 --out "$OUT"
echo "[pod2] $(date +%T) x4 done"
python retention/mech_pod2.py --task x6 --out "$OUT"
echo "[pod2] $(date +%T) all done"
