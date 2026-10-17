#!/usr/bin/env bash
# P1 main-run driver (one model per pod, pod kept alive for the whole run). Serves the model ONCE as an
# OpenAI-compatible vLLM endpoint, then runs every trajectory in JOBS in parallel against it, each as its
# own EC harness process (ext/ec_repo, untrusted, executed only on the pod per card).
#
# Args: MODEL SERVED JOBS_TSV OUTROOT MAXTOK ROUNDS RPARSER TOOLPARSER
# Env:  MAXLEN (default 131072, capped to the model's max_position_embeddings)
#       THINK=1 adds --default-chat-template-kwargs '{"enable_thinking": true}' when vLLM supports it
# JOBS_TSV lines: <arm> <TAB> <rep> <TAB> <kind: live|compliant|violating> <TAB> <sequence dir>
#   live      = EC as published (both agents are the served model)
#   compliant = scripted peer: raw-log prefix messages + REJECT   (published cache, no model calls for Bob)
#   violating = scripted peer: own-summary messages + ACCEPT      (published cache)
set -uo pipefail
MODEL="$1"; SERVED="$2"; JOBS="$3"; OUTROOT="$4"; MAXTOK="$5"; ROUNDS="$6"; RPARSER="$7"; TOOLPARSER="$8"
case "$JOBS" in /*) ;; *) JOBS="$(pwd)/$JOBS" ;; esac   # absolute: step 8 runs after cd into the harness dir
[ -f "$JOBS" ] || { echo "[arms] FATAL jobs file not found: $JOBS"; exit 29; }
PORT=8000
POD=/root/NEEL/mas_pilot/p1_baserate/pod
ECDIR=/root/NEEL/mas_pilot/p1_baserate/ext/ec_repo
ABS_OUT=/root/NEEL/$OUTROOT
PEER_ROOT=/root/peer_dl            # untrusted download, its own dir outside the repo
mkdir -p "$ABS_OUT"
export PATH="$HOME/.local/bin:$PATH"
echo "[arms] model=$MODEL served=$SERVED jobs=$JOBS maxtok=$MAXTOK rounds=$ROUNDS rparser=$RPARSER toolparser=$TOOLPARSER maxlen=${MAXLEN:-131072} think=${THINK:-0} gpu_util=${GPU_UTIL:-0.92}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. vLLM venv (+ninja)
bash "$POD/setup_vllm.sh" || { echo "[arms] FATAL vllm setup failed"; exit 20; }

# 2. HF token (never printed), ninja on PATH, native sampler
if [ -z "${HF_TOKEN:-}" ]; then HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^HF_TOKEN=//p'); export HF_TOKEN; fi
[ -n "${HF_TOKEN:-}" ] && echo "[arms] HF_TOKEN present (len ${#HF_TOKEN})" || echo "[arms] WARN: HF_TOKEN absent (gated models will fail)"
(apt-get install -y -qq ninja-build >/dev/null 2>&1 || (apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq ninja-build >/dev/null 2>&1)) || true
export PATH="/root/vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
command -v ninja >/dev/null && echo "[arms] ninja $(ninja --version)" || echo "[arms] WARN: ninja not on PATH"

# 3. Parsers (auto = resolve from the installed vLLM source)
if [ "$RPARSER" = "auto" ] || [ "$TOOLPARSER" = "auto" ]; then
  read -r RP_AUTO TP_AUTO <<< "$(/root/vllm/bin/python "$POD/resolve_parsers.py" "$MODEL")"
  [ "$RPARSER" = "auto" ] && RPARSER="$RP_AUTO"
  [ "$TOOLPARSER" = "auto" ] && TOOLPARSER="$TP_AUTO"
fi
echo "[arms] parsers: reasoning=$RPARSER tool=$TOOLPARSER"
if [ "$RPARSER" = "NONE" ] || [ "$TOOLPARSER" = "NONE" ] || [ -z "$RPARSER" ] || [ -z "$TOOLPARSER" ]; then
  echo "[arms] FATAL no matching parser for $MODEL"; exit 26; fi

# 4. Context length: requested MAXLEN capped to the model's own limit
MAXPOS=$(/root/vllm/bin/python - "$MODEL" 2>/dev/null <<'PY'
import sys
from transformers import AutoConfig
c = AutoConfig.from_pretrained(sys.argv[1])
t = c.get_text_config() if hasattr(c, "get_text_config") else c
print(int(getattr(t, "max_position_embeddings", 0) or 0))
PY
)
MAXLEN_REQ=${MAXLEN:-131072}
if [ -n "$MAXPOS" ] && [ "$MAXPOS" -gt 0 ] 2>/dev/null && [ "$MAXPOS" -lt "$MAXLEN_REQ" ]; then MAXLEN_EFF=$MAXPOS; else MAXLEN_EFF=$MAXLEN_REQ; fi
echo "[arms] model max_position_embeddings=${MAXPOS:-unknown}; serving max-model-len=$MAXLEN_EFF"

# 5. Thinking: enable via the chat template when requested and supported
# CPU KV offloading (vLLM native OffloadingConnector): evicted prefix-cache blocks go to pinned host RAM and
# come back by DMA instead of being recomputed. Same bf16 KV, so outputs are unchanged; only re-prefill is saved.
OFFLOAD_ARGS=()
if [ -n "${KV_OFFLOAD_GB:-}" ]; then
  if [ "$KV_OFFLOAD_GB" = "auto" ]; then
    avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    KV_OFFLOAD_GB=$(( avail_gb * 55 / 100 )); [ "$KV_OFFLOAD_GB" -gt 96 ] && KV_OFFLOAD_GB=96
    echo "[arms] host MemAvailable ${avail_gb} GiB -> KV offload ${KV_OFFLOAD_GB} GiB"
  fi
  [ "$KV_OFFLOAD_GB" -ge 4 ] 2>/dev/null && OFFLOAD_ARGS=(--kv-offloading-size "$KV_OFFLOAD_GB" --kv-offloading-backend native)
fi
MM_ARGS=()
if [ -n "${MM_LIMIT:-}" ]; then MM_ARGS=(--limit-mm-per-prompt "$MM_LIMIT"); echo "[arms] multimodal limit: $MM_LIMIT"; fi
THINK_ARGS=()
if [ "${THINK:-0}" = "1" ]; then
  if /root/vllm/bin/python -m vllm.entrypoints.openai.api_server --help 2>/dev/null | grep -q -- "--default-chat-template-kwargs"; then
    THINK_ARGS=(--default-chat-template-kwargs '{"enable_thinking": true}')
    echo "[arms] thinking: enable_thinking=true via --default-chat-template-kwargs"
  else
    echo "[arms] WARN: vLLM lacks --default-chat-template-kwargs; thinking left at the template default"
  fi
fi

start_server() {   # $1 = mode label, rest = extra vLLM flags; returns 0 when healthy
  local label="$1"; shift
  echo "[arms] starting vLLM ($label) at $(date -u +%H:%M:%S)"
  nohup /root/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
    --dtype bfloat16 --max-model-len "$MAXLEN_EFF" --gpu-memory-utilization "${GPU_UTIL:-0.92}" --seed 0 "$@" \
    --reasoning-parser "$RPARSER" --enable-auto-tool-choice --tool-call-parser "$TOOLPARSER" "${THINK_ARGS[@]}" "${MM_ARGS[@]}" "${OFFLOAD_ARGS[@]}" \
    >> "$ABS_OUT/vllm_server.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "[arms] server healthy after ${i}0s at $(date -u +%H:%M:%S) (mode: $label)"; SERVE_MODE="$label"; return 0; fi
    if ! kill -0 $SRV 2>/dev/null; then
      echo "[arms] server died (mode: $label); last errors:"
      grep -E "Error|error:|ValueError|RuntimeError|larger than|KV cache" "$ABS_OUT/vllm_server.log" | tail -6
      if [ ${#OFFLOAD_ARGS[@]} -gt 0 ] && tail -120 "$ABS_OUT/vllm_server.log" | grep -qiE "offload|kv.transfer|kv_connector|OffloadingConnector"; then
        echo "[arms] KV offloading rejected for this model; retrying $label WITHOUT offloading"; OFFLOAD_ARGS=(); sleep 5; start_server "$label" "$@"; return $?; fi
      if [ ${#MM_ARGS[@]} -gt 0 ] && tail -80 "$ABS_OUT/vllm_server.log" | grep -qiE "limit.mm.per.prompt|modalit"; then
        echo "[arms] multimodal limit rejected; retrying $label without it"; MM_ARGS=(); sleep 5; start_server "$label" "$@"; return $?; fi
      return 1; fi
    sleep 10
  done
  echo "[arms] server never became healthy (mode: $label)"; kill $SRV 2>/dev/null; return 1
}
T_SERVE0=$(date +%s)
# bf16 KV first (compiled, then eager); fp8 KV only if bf16 cannot hold one max-length request.
start_server "compiled/bf16-kv" \
  || { sleep 5; start_server "eager/bf16-kv" --enforce-eager; } \
  || { sleep 5; start_server "compiled/fp8-kv" --kv-cache-dtype fp8; } \
  || { sleep 5; start_server "eager/fp8-kv" --enforce-eager --kv-cache-dtype fp8; } \
  || { echo "[arms] FATAL server failed in every mode"; exit 21; }
echo "[arms] serve-ready seconds (download+load+warmup): $(( $(date +%s) - T_SERVE0 ))  mode=$SERVE_MODE"
echo "[arms] unauthenticated-HF warnings in server log: $(grep -c 'unauthenticated requests' "$ABS_OUT/vllm_server.log")"
echo "[arms] KV offloading: ${OFFLOAD_ARGS[*]:-off}; gpu_util=${GPU_UTIL:-0.92}"
grep -iE "OffloadingConnector|offloading" "$ABS_OUT/vllm_server.log" | sed -E 's/^.*\] /[arms] /' | head -3
grep -E "GPU KV cache size|Maximum concurrency|Model loading took" "$ABS_OUT/vllm_server.log" | sed -E 's/^.*\] /[arms] /' | tail -3

# 5b. Optional context-budget check (does vLLM reject prompt + max_tokens > max-model-len, as the harness needs?)
if [ "${CTX_CHECK:-0}" = "1" ]; then
  /root/vllm/bin/python "$POD/ctx_check.py" "http://localhost:$PORT" "$SERVED" "$MAXLEN_EFF" "$MAXTOK" || echo "[arms] WARN ctx_check failed"
fi

# 6. Harness venv
uv venv -q /root/ecenv --python 3.12 || { echo "[arms] FATAL ecenv create failed"; exit 23; }
VIRTUAL_ENV=/root/ecenv uv pip install -q -r "$ECDIR/requirements.txt" || { echo "[arms] FATAL ec deps install failed"; exit 24; }

# 7. Published peer cache (only the reps the scripted-peer jobs use)
PEER_REPS=$(awk -F'\t' '$3=="compliant"||$3=="violating"{printf "%03d\n",$2}' "$JOBS" | sort -u | tr '\n' ' ')
if [ -n "$PEER_REPS" ]; then
  echo "[arms] fetching published peer cache reps: $PEER_REPS"
  /root/vllm/bin/python - "$PEER_ROOT" $PEER_REPS <<'PY' || { echo "[arms] FATAL peer cache download failed"; exit 27; }
import sys
from huggingface_hub import snapshot_download
root, reps = sys.argv[1], sys.argv[2:]
snapshot_download("SALT-NLP/agent-collusion", repo_type="dataset", local_dir=root,
                  allow_patterns=[f"peer-cache/rep{r}/*.json" for r in reps])
print("[arms] peer cache ok:", root)
PY
  for r in $PEER_REPS; do echo "[arms]   rep$r: $(ls "$PEER_ROOT/peer-cache/rep$r" 2>/dev/null | wc -l) files"; done
fi

# 8. Launch every trajectory in parallel
cd "$ECDIR" || exit 25
export OPENAI_API_BASE="http://localhost:$PORT/v1"
export OPENAI_API_KEY="sk-local-dummy"
COMMON=(--alice-model "openai/$SERVED" --alice-reasoning-effort default --alice-temperature 1.0
        --alice-max-output-tokens "$MAXTOK" --max-rounds "$ROUNDS" --repeats 1)
declare -a PIDS NAMES LOGS
T_RUN0=$(date +%s)
while IFS=$'\t' read -r ARM REP KIND SEQ; do
  [ -z "${ARM:-}" ] && continue
  case "$ARM" in \#*) continue;; esac
  mkdir -p "$ABS_OUT/$ARM"
  case "$KIND" in
    live)      BOB=(--bob-model "openai/$SERVED" --bob-reasoning-effort default --bob-temperature 1.0 --bob-max-output-tokens "$MAXTOK") ;;
    compliant) BOB=(--bob-model controlled --controlled-bob-cache "$PEER_ROOT/peer-cache" --controlled-bob-messages raw-prefix --controlled-bob-verdict reject) ;;
    violating) BOB=(--bob-model controlled --controlled-bob-cache "$PEER_ROOT/peer-cache" --controlled-bob-messages summary --controlled-bob-verdict accept) ;;
    *) echo "[arms] FATAL unknown kind $KIND"; exit 28 ;;
  esac
  LOG="$ABS_OUT/$ARM/rep$(printf %03d "$REP").log"
  /root/ecenv/bin/python -m experiments "${COMMON[@]}" "${BOB[@]}" --task-sequence-record "$SEQ" \
    --start-index "$REP" --output-dir "$ABS_OUT/$ARM" > "$LOG" 2>&1 &
  PIDS+=($!); NAMES+=("$ARM/rep$REP"); LOGS+=("$LOG")
  echo "[arms] launched $ARM rep$REP ($KIND) pid $! at $(date -u +%H:%M:%S)"
  sleep 2
done < "$JOBS"
echo "[arms] ${#PIDS[@]} trajectories running"

# 9. Heartbeat every 5 min (keeps the launcher's stall guard quiet; milestones for the coordinator)
declare -A DONE
while :; do
  running=0
  for i in "${!PIDS[@]}"; do
    if [ -z "${DONE[$i]:-}" ]; then
      if kill -0 "${PIDS[$i]}" 2>/dev/null; then running=$((running+1));
      else wait "${PIDS[$i]}"; DONE[$i]=$?; echo "[arms] DONE ${NAMES[$i]} rc=${DONE[$i]} at $(date -u +%H:%M:%S) after $(( ($(date +%s)-T_RUN0)/60 )) min"; fi
    fi
  done
  [ "$running" -eq 0 ] && break
  if [ -n "${HARNESS_MAX_MIN:-}" ] && [ $(( ($(date +%s)-T_RUN0)/60 )) -ge "$HARNESS_MAX_MIN" ]; then
    echo "[arms] HARNESS_MAX_MIN=$HARNESS_MAX_MIN reached; stopping $running running trajectories (smoke/time cap)"
    for i in "${!PIDS[@]}"; do [ -z "${DONE[$i]:-}" ] && pkill -TERM -P "${PIDS[$i]}" 2>/dev/null; [ -z "${DONE[$i]:-}" ] && kill -TERM "${PIDS[$i]}" 2>/dev/null && DONE[$i]=stopped; done
    break
  fi
  prog=""
  for i in "${!PIDS[@]}"; do
    n=$(grep -c "Saved incremental results after" "${LOGS[$i]}" 2>/dev/null); prog+=" ${NAMES[$i]}:${n}ep"
  done
  ctx=$(grep -c "context limit: dropped" "${LOGS[@]}" 2>/dev/null | awk -F: '{s+=$NF} END{print s+0}')
  errs=$(grep -lE "Traceback|LLMRetryLimitExceeded" "${LOGS[@]}" 2>/dev/null | wc -l)
  met=$(grep -E "Avg prompt throughput" "$ABS_OUT/vllm_server.log" | tail -1 | sed -E 's/^.*Avg prompt/prompt/; s/ tokens\/s//g')
  pre=$(grep -ci "preempt" "$ABS_OUT/vllm_server.log")
  echo "[arms] $(date -u +%H:%M) +$(( ($(date +%s)-T_RUN0)/60 ))m running=$running |$prog | ctx-prunes=$ctx tracebacks=$errs preempt-lines=$pre | $met"
  sleep "${HEARTBEAT_S:-300}"
done
echo "[arms] all trajectories finished; harness minutes: $(( ($(date +%s)-T_RUN0)/60 ))"
fails=0; for i in "${!PIDS[@]}"; do [ "${DONE[$i]:-}" != "0" ] && fails=$((fails+1)); done
for i in "${!PIDS[@]}"; do n=$(grep -c "Saved incremental results after" "${LOGS[$i]}" 2>/dev/null); echo "[arms] final ${NAMES[$i]}: status=${DONE[$i]:-?} episodes_saved=$n"; done
echo "[arms] failed trajectories: $fails of ${#PIDS[@]}"
kill $SRV 2>/dev/null
exit 0
