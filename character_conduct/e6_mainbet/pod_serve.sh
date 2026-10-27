#!/usr/bin/env bash
# CC-E6 stage 1 pod (DESIGN.md §3-4; one A100 SXM4 80GB). Launch with launch_serve.sh, never by hand.
#   1. vLLM setup; inputs sha256; Qwen3-14B @40c06982
#   2. hint_reads.py: M0's Hint-MMLU behaviour (offline vLLM) -> OUT/hint_beh.jsonl
#   3. vllm serve M0 on 127.0.0.1:8000 (bf16, 40,960 tokens, hermes tool parser, LoRA enabled for later stages) for the
#      laptop's ODCV episodes through an ssh tunnel; runs until /root/STOP exists or T0 + MAXMIN - 6 min
# env: T0, MAXMIN, OUT (required); PY (/root/venv/bin/python), VPY (/root/vllm/bin/python)
set -uo pipefail
cd "$(dirname "$0")/../.."
H=character_conduct/e6_mainbet
: "${T0:?set T0}" "${MAXMIN:?set MAXMIN}" "${OUT:?set OUT}"
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
DEADLINE=$(( T0 + (MAXMIN - 6) * 60 ))
mkdir -p "$OUT"; LOG=$OUT/pod.log
export VLLM_USE_FLASHINFER_SAMPLER=0
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG"
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log" | tee -a "$LOG"; exit 1; }
"$PY" - <<PY 2>&1 | tee -a "$LOG"
import hashlib; from pathlib import Path
want = {"character_conduct/e1b_coupling/data/items.jsonl.gz": "c3aed287cdff3b1709d58cb1eef7a41c71a9018912c3669c13388550d1ad17a6"}
bad = [p for p, s in want.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest() != s]
assert not bad, f"inputs changed: {bad}"; print("[e6] inputs sha256 ok")
PY
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
M=$("$PY" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-14B', revision='40c069824f4251a91eefaf281ebe4c544efd3e18'))" 2>>"$LOG")
[ -d "$M" ] || { echo "[e6] base snapshot failed" | tee -a "$LOG"; exit 1; }
"$VPY" -m character_conduct.e6_mainbet.hint_reads --model "$M" --out "$OUT" 2>&1 | grep --line-buffered -v "it/s\]" | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "[e6] hint reads failed" | tee -a "$LOG"; exit 1; }
"$VPY" -m vllm.entrypoints.openai.api_server --model "$M" --served-model-name qwen3-14b --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 40960 --gpu-memory-utilization 0.9 --enable-auto-tool-choice --tool-call-parser hermes \
  --enable-prefix-caching --max-num-seqs 64 --enable-lora --max-lora-rank 512 --max-loras 2 --seed 0 \
  > "$OUT/serve.log" 2>&1 &
SRV=$!
for i in $(seq 1 120); do curl -s 127.0.0.1:8000/v1/models >/dev/null 2>&1 && break; sleep 5; done
curl -s 127.0.0.1:8000/v1/models | head -c 300 | tee -a "$LOG"; echo | tee -a "$LOG"
echo "[e6] serving; touch /root/STOP to end" | tee -a "$LOG"
while [ ! -f /root/STOP ] && [ "$(date +%s)" -lt "$DEADLINE" ] && kill -0 "$SRV" 2>/dev/null; do sleep 15; done
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null
grep -c "POST /v1/chat/completions" "$OUT/serve.log" | sed 's/^/[e6] chat requests served: /' | tee -a "$LOG"
echo "[e6] done" | tee -a "$LOG"
exit 0
