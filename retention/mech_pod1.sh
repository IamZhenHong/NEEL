#!/usr/bin/env bash
# Pod 1 of prereg retention/prereg/retention_mechanisms_v1.md (forward passes + the reduced Assistant Axis pipeline; no API).
# Phase A (vLLM): axis generations for Qwen2.5-7B, Qwen3-32B, Qwen2.5-32B; then the open judge (Qwen2.5-32B, same server) scores
# all role responses. Phase B (HF): mech_pod1.py captures for q7b, q32b, qwen3_32b.
#   uv run bash retention/mech_pod1.sh            (SMOKE=1 for a tiny end-to-end check)
set -euo pipefail
OUT=retention/results/retention_mechanisms/pod1
mkdir -p "$OUT/axis_gen" /root/scratch
SPID=""

gpu_free() {   # wait until the GPU holds < 3 GB (vLLM engine processes exited)
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${used:-0}" -lt 3000 ] && return 0; sleep 5
  done
  echo "[pod1] GPU still holds ${used} MiB"; nvidia-smi; return 1
}
serve() {   # MODEL PORT
  echo "[pod1] $(date +%T) serving $1"
  bash retention/vllm_serve.sh "$1" "$2" 0.90 4096 > "/root/vllm_$(basename "$1").log" 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$2/health" >/dev/null && { echo "[pod1] $(date +%T) up: $1"; return 0; }
    kill -0 "$SPID" 2>/dev/null || { echo "[pod1] vllm exited"; tail -60 "/root/vllm_$(basename "$1").log"; return 1; }
    sleep 10
  done
  echo "[pod1] vllm did not come up in 30 min"; tail -60 "/root/vllm_$(basename "$1").log"; return 1
}
stop() {
  [ -n "$SPID" ] && { kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }
  pkill -f "vllm serve" 2>/dev/null || true; SPID=""; gpu_free
}
trap 'stop || true' EXIT

# ---- Phase A: generations (resumable files) ----
serve Qwen/Qwen2.5-7B-Instruct 8000
python retention/mech_axis_gen.py --model Qwen/Qwen2.5-7B-Instruct --url http://127.0.0.1:8000 --out "$OUT/axis_gen/q7b.jsonl" --turn1 q7b
stop
serve Qwen/Qwen3-32B 8000
python retention/mech_axis_gen.py --model Qwen/Qwen3-32B --url http://127.0.0.1:8000 --out "$OUT/axis_gen/qwen3_32b.jsonl"
stop
serve Qwen/Qwen2.5-32B-Instruct 8000
python retention/mech_axis_gen.py --model Qwen/Qwen2.5-32B-Instruct --url http://127.0.0.1:8000 --out "$OUT/axis_gen/q32b.jsonl" --turn1 q32b
python retention/mech_axis_judge.py --gen "$OUT/axis_gen/q7b.jsonl" --gen "$OUT/axis_gen/qwen3_32b.jsonl" --gen "$OUT/axis_gen/q32b.jsonl" \
  --url http://127.0.0.1:8000 --judge Qwen/Qwen2.5-32B-Instruct --out "$OUT/axis_scores.jsonl"
for f in /root/vllm_*.log; do cp "$f" "$OUT/" || true; done
stop
echo "[pod1] $(date +%T) phase A done"

# ---- Phase B: HF captures ----
python retention/mech_pod1.py --model q7b --tasks x1,x2,x3,x7,x7free,axis --out "$OUT" \
  --axis-gen "$OUT/axis_gen/q7b.jsonl" --axis-scores "$OUT/axis_scores.jsonl"
python retention/mech_pod1.py --model q32b --tasks x1,x2,x3,x7,axis --out "$OUT" \
  --axis-gen "$OUT/axis_gen/q32b.jsonl" --axis-scores "$OUT/axis_scores.jsonl"
python retention/mech_pod1.py --model qwen3_32b --tasks axis --out "$OUT" \
  --axis-gen "$OUT/axis_gen/qwen3_32b.jsonl" --axis-scores "$OUT/axis_scores.jsonl"
echo "[pod1] $(date +%T) all done"
