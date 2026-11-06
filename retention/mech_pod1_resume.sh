#!/usr/bin/env bash
# Resume of Pod 1 (prereg retention/prereg/retention_mechanisms_v1.md) after the 2026-11-06 run lost X3 onwards: the launcher's
# ship step drops every results/ path, so the long conversations never reached the pod (X1/X2 for q7b and all axis generations +
# judge scores completed and are on the laptop). Inputs now travel as retention/data/mech_inputs_v1.tar.gz. Remaining work:
# turn-1 default replies (V8b) for q7b/q32b, then q7b x3,x7,x7free,axis; q32b x1,x2,x3,x7,axis; qwen3_32b axis.
set -euo pipefail
cd /root/NEEL
tar -xzf retention/data/mech_inputs_v1.tar.gz -C /root/NEEL
OUT=retention/results/retention_mechanisms/pod1
n7=$(ls retention/results/retention_longconv/q7b/convs/*.json | wc -l); n32=$(ls retention/results/retention_longconv/q32b/convs/*.json | wc -l)
echo "[resume] conversations on pod: q7b $n7, q32b $n32"; [ "$n7" -eq 160 ] && [ "$n32" -eq 160 ] || { echo "[resume] missing conversations"; exit 1; }
mkdir -p /root/scratch
SPID=""
gpu_free() {
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${used:-0}" -lt 3000 ] && return 0; sleep 5
  done
  echo "[resume] GPU still holds ${used} MiB"; return 1
}
serve() {
  echo "[resume] $(date +%T) serving $1"
  bash retention/vllm_serve.sh "$1" "$2" 0.90 4096 > "/root/vllm_$(basename "$1").log" 2>&1 &
  SPID=$!
  for i in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$2/health" >/dev/null && { echo "[resume] $(date +%T) up: $1"; return 0; }
    kill -0 "$SPID" 2>/dev/null || { echo "[resume] vllm exited"; tail -60 "/root/vllm_$(basename "$1").log"; return 1; }
    sleep 10
  done
  echo "[resume] vllm did not come up in 30 min"; return 1
}
stop() {
  [ -n "$SPID" ] && { kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }
  pkill -f "vllm serve" 2>/dev/null || true; SPID=""; gpu_free
}
trap 'stop || true' EXIT

serve Qwen/Qwen2.5-7B-Instruct 8000
python retention/mech_axis_gen.py --model Qwen/Qwen2.5-7B-Instruct --url http://127.0.0.1:8000 --out "$OUT/axis_gen/q7b.jsonl" --turn1 q7b
stop
serve Qwen/Qwen2.5-32B-Instruct 8000
python retention/mech_axis_gen.py --model Qwen/Qwen2.5-32B-Instruct --url http://127.0.0.1:8000 --out "$OUT/axis_gen/q32b.jsonl" --turn1 q32b
stop
echo "[resume] $(date +%T) turn-1 generations done"

python retention/mech_pod1.py --model q7b --tasks x3,x7,x7free,axis --out "$OUT" \
  --axis-gen "$OUT/axis_gen/q7b.jsonl" --axis-scores "$OUT/axis_scores.jsonl"
python retention/mech_pod1.py --model q32b --tasks x1,x2,x3,x7,axis --out "$OUT" \
  --axis-gen "$OUT/axis_gen/q32b.jsonl" --axis-scores "$OUT/axis_scores.jsonl"
python retention/mech_pod1.py --model qwen3_32b --tasks axis --out "$OUT" \
  --axis-gen "$OUT/axis_gen/qwen3_32b.jsonl" --axis-scores "$OUT/axis_scores.jsonl"
echo "[pod1] $(date +%T) all done"
