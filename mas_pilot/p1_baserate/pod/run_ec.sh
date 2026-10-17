#!/usr/bin/env bash
# P1 EC harness driver on a RunPod A100. Serves ONE self-hosted model as an OpenAI-compatible
# vLLM endpoint and points the Emergent Collusion harness (ext/ec_repo, untrusted data, run only
# on the pod per card) at localhost. Fail-soft: capture diagnostics, never leave the pod hung.
#
# Args: MODEL SERVED PORT MANIFEST_DIR OUTDIR MAXTOK ROUNDS RPARSER TOOLPARSER REPEATS EFFORT [EXTRA...]
set -uo pipefail
MODEL="$1"; SERVED="$2"; PORT="${3:-8000}"; MANIFEST_DIR="$4"; OUTDIR="$5"
MAXTOK="${6:-6000}"; ROUNDS="${7:-3}"; RPARSER="${8:-qwen3}"; TOOLPARSER="${9:-hermes}"
REPEATS="${10:-1}"; EFFORT="${11:-default}"; shift 11 || true; EXTRA=("$@")
ECDIR=/root/NEEL/mas_pilot/p1_baserate/ext/ec_repo
ABS_OUT=/root/NEEL/$OUTDIR
mkdir -p "$ABS_OUT"
export PATH="$HOME/.local/bin:$PATH"
echo "[ec] model=$MODEL served=$SERVED port=$PORT maxtok=$MAXTOK rounds=$ROUNDS rparser=$RPARSER toolparser=$TOOLPARSER repeats=$REPEATS effort=$EFFORT extra=${EXTRA[*]:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. vLLM venv (its own torch); vendored copy so this card does not depend on other folders.
bash /root/NEEL/mas_pilot/p1_baserate/pod/setup_vllm.sh || { echo "[ec] FATAL vllm setup failed"; exit 20; }

# 2. Serve the model (OpenAI-compatible).
# 2a. HF token: the launcher exports it when it is started with HF_TOKEN set; fall back to the pod env.
#     Never print the value, only whether it is present (gemma-4-31B-it is gated).
if [ -z "${HF_TOKEN:-}" ]; then
  HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^HF_TOKEN=//p'); export HF_TOKEN
fi
[ -n "${HF_TOKEN:-}" ] && echo "[ec] HF_TOKEN present (len ${#HF_TOKEN})" || echo "[ec] WARN: HF_TOKEN absent (gated models will fail)"
# 2b. FlashInfer JIT-compiles its top-k/top-p sampling kernel at warm-up and shells out to `ninja`.
#     Install a system ninja (most robust) and keep the venv ninja on PATH; also use vLLM's native
#     PyTorch sampler (no JIT, identical outputs) as a belt-and-braces fallback.
(apt-get install -y -qq ninja-build >/dev/null 2>&1 || (apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq ninja-build >/dev/null 2>&1)) || true
export PATH="/root/vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
command -v ninja >/dev/null && echo "[ec] ninja $(ninja --version) at $(command -v ninja)" || echo "[ec] WARN: ninja not on PATH"

# 2c. Parser names: "auto" resolves from vLLM's own registries (scanned from the installed source, so it
#     does not depend on a registry API), preferring the model family's dedicated parser.
if [ "$RPARSER" = "auto" ] || [ "$TOOLPARSER" = "auto" ]; then
  read -r RP_AUTO TP_AUTO <<< "$(/root/vllm/bin/python /root/NEEL/mas_pilot/p1_baserate/pod/resolve_parsers.py "$MODEL")"
  [ "$RPARSER" = "auto" ] && RPARSER="$RP_AUTO"
  [ "$TOOLPARSER" = "auto" ] && TOOLPARSER="$TP_AUTO"
  echo "[ec] resolved parsers: reasoning=$RPARSER tool=$TOOLPARSER"
  if [ "$RPARSER" = "NONE" ] || [ "$TOOLPARSER" = "NONE" ] || [ -z "$RPARSER" ] || [ -z "$TOOLPARSER" ]; then
    echo "[ec] FATAL no matching parser for $MODEL"; exit 26; fi
fi

start_server() {   # $1 = extra vLLM flags (e.g. --enforce-eager); returns 0 when healthy
  echo "[ec] starting vLLM server ${1:-(compiled)} at $(date -u +%H:%M:%S)"
  nohup /root/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
    --dtype bfloat16 --max-model-len "${MAXLEN:-40960}" --gpu-memory-utilization 0.92 --seed 0 $1 \
    --reasoning-parser "$RPARSER" --enable-auto-tool-choice --tool-call-parser "$TOOLPARSER" \
    >> "$ABS_OUT/vllm_server.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "[ec] server healthy after ${i}0s at $(date -u +%H:%M:%S) (mode: ${1:-compiled})"; return 0; fi
    if ! kill -0 $SRV 2>/dev/null; then echo "[ec] server process died (mode: ${1:-compiled}); tail:"; tail -25 "$ABS_OUT/vllm_server.log"; return 1; fi
    sleep 10
  done
  echo "[ec] server never became healthy (mode: ${1:-compiled})"; kill $SRV 2>/dev/null; return 1
}
T_SERVE0=$(date +%s)
# Compiled first (representative throughput for the main run); eager fallback reuses the cached weights.
start_server "" || { sleep 5; start_server "--enforce-eager"; } || { echo "[ec] FATAL server failed in both modes"; exit 21; }
echo "[ec] serve-ready seconds (download+load+warmup): $(( $(date +%s) - T_SERVE0 ))"
grep -c "unauthenticated requests" "$ABS_OUT/vllm_server.log" | sed 's/^/[ec] unauthenticated-HF warnings in server log: /'
curl -s "http://localhost:$PORT/v1/models" | head -c 500; echo

# 3. Harness venv from the repo's pinned requirements (installed into a separate venv, not run from ext/).
uv venv -q /root/ecenv --python 3.12 || { echo "[ec] FATAL ecenv create failed"; exit 23; }
VIRTUAL_ENV=/root/ecenv uv pip install -q -r "$ECDIR/requirements.txt" || { echo "[ec] FATAL ec deps install failed"; exit 24; }

# 4. Drive the harness (cwd = repo per its cwd-relative task/results paths; sanctioned on-pod execution).
cd "$ECDIR" || exit 25
export OPENAI_API_BASE="http://localhost:$PORT/v1"
export OPENAI_API_KEY="sk-local-dummy"
T_RUN0=$(date +%s)
echo "[ec] launching experiments at $(date -u +%H:%M:%S)"
/root/ecenv/bin/python -m experiments \
  --alice-model "openai/$SERVED" --bob-model "openai/$SERVED" \
  --alice-reasoning-effort "$EFFORT" --bob-reasoning-effort "$EFFORT" \
  --alice-max-output-tokens "$MAXTOK" --bob-max-output-tokens "$MAXTOK" \
  --alice-temperature 1.0 --bob-temperature 1.0 \
  --max-rounds "$ROUNDS" \
  --task-sequence-record "$MANIFEST_DIR" \
  --repeats "$REPEATS" \
  --output-dir "$ABS_OUT" "${EXTRA[@]}"
RC=$?
echo "[ec] experiments exit rc=$RC at $(date -u +%H:%M:%S); harness seconds: $(( $(date +%s) - T_RUN0 ))"

# 5. Collect run.json(s) + usage journal into OUTDIR (already there) and tail server log.
find "$ABS_OUT" -maxdepth 3 -name 'run.json' -print 2>/dev/null | head
echo "[ec] GPU after run:"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
kill $SRV 2>/dev/null
exit $RC
