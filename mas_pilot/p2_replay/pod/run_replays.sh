#!/usr/bin/env bash
# P2 replay driver (one model per pod). Serves the model ONCE (same serving as P1: parsers, thinking, ninja,
# CPU KV offload sized to /dev/shm), verifies every seed record's restored state, then resamples single
# episodes with experiments.runner --resume-from, CONC at a time.
#   phase 1: SMOKE jobs (all lines of SMOKE_TSV)
#   phase 2: MAIN jobs, only after /root/p2_go appears (its content = resamples R; jobs with r >= R skipped);
#            waits up to GO_TIMEOUT_MIN, else exits (so p2-freeze can be tagged between smoke and main)
# Args: MODEL SERVED SMOKE_TSV MAIN_TSV OUTROOT MAXTOK RPARSER TOOLPARSER
# Jobs TSV: <record.json repo-rel> TAB <out subdir> TAB <resample index>
# Env: MAXLEN THINK GPU_UTIL KV_OFFLOAD_GB MM_LIMIT CONC (default 3) GO_TIMEOUT_MIN (default 25) HEARTBEAT_S
set -uo pipefail
MODEL="$1"; SERVED="$2"; SMOKE="$3"; MAINJ="$4"; OUTROOT="$5"; MAXTOK="$6"; RPARSER="$7"; TOOLPARSER="$8"
for v in SMOKE MAINJ; do eval "f=\$$v"; case "$f" in /*|none) ;; *) eval "$v=\"$(pwd)/$f\"" ;; esac; done
PORT=8000
POD=/root/NEEL/mas_pilot/p2_replay/pod
ECDIR=/root/NEEL/mas_pilot/p1_baserate/ext/ec_repo
ABS_OUT=/root/NEEL/$OUTROOT
PEER_ROOT=/root/peer_dl            # same path P1 used, so the seeds' recorded cache path matches
mkdir -p "$ABS_OUT"
export PATH="$HOME/.local/bin:$PATH"
echo "[p2] model=$MODEL served=$SERVED smoke=$SMOKE main=$MAINJ maxtok=$MAXTOK conc=${CONC:-3} maxlen=${MAXLEN:-131072} think=${THINK:-0} gpu_util=${GPU_UTIL:-0.92}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. vLLM venv (+ninja)
bash "$POD/setup_vllm.sh" || { echo "[p2] FATAL vllm setup failed"; exit 20; }

# 2. HF token (never printed), ninja on PATH, native sampler
if [ -z "${HF_TOKEN:-}" ]; then HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^HF_TOKEN=//p'); export HF_TOKEN; fi
[ -n "${HF_TOKEN:-}" ] && echo "[p2] HF_TOKEN present (len ${#HF_TOKEN})" || echo "[p2] WARN: HF_TOKEN absent (gated models will fail)"
(apt-get install -y -qq ninja-build >/dev/null 2>&1 || (apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq ninja-build >/dev/null 2>&1)) || true
export PATH="/root/vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
command -v ninja >/dev/null && echo "[p2] ninja $(ninja --version)" || echo "[p2] WARN: ninja not on PATH"

# 3. Parsers (auto = resolve from the installed vLLM source)
if [ "$RPARSER" = "auto" ] || [ "$TOOLPARSER" = "auto" ]; then
  read -r RP_AUTO TP_AUTO <<< "$(/root/vllm/bin/python "$POD/resolve_parsers.py" "$MODEL")"
  [ "$RPARSER" = "auto" ] && RPARSER="$RP_AUTO"
  [ "$TOOLPARSER" = "auto" ] && TOOLPARSER="$TP_AUTO"
fi
echo "[p2] parsers: reasoning=$RPARSER tool=$TOOLPARSER"
if [ "$RPARSER" = "NONE" ] || [ "$TOOLPARSER" = "NONE" ] || [ -z "$RPARSER" ] || [ -z "$TOOLPARSER" ]; then
  echo "[p2] FATAL no matching parser for $MODEL"; exit 26; fi

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
echo "[p2] model max_position_embeddings=${MAXPOS:-unknown}; serving max-model-len=$MAXLEN_EFF"

# 5. Thinking: enable via the chat template when requested and supported
# CPU KV offloading (vLLM native OffloadingConnector): evicted prefix-cache blocks go to pinned host RAM and
# come back by DMA instead of being recomputed. Same bf16 KV, so outputs are unchanged; only re-prefill is saved.
OFFLOAD_ARGS=()
if [ -n "${KV_OFFLOAD_GB:-}" ]; then
  if [ "$KV_OFFLOAD_GB" = "auto" ]; then
    # The pinned CPU tier lives in /dev/shm, which is the real limit in RunPod containers (2026-10-17: 96 GiB
    # requested vs 54 GiB free shm made vLLM refuse offloading). Size = min(55% RAM, 80% free shm, 96 GiB).
    avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
    shm_gb=$(df -m /dev/shm | awk 'NR==2{printf "%d", $4/1024}')
    KV_OFFLOAD_GB=$(( avail_gb * 55 / 100 ))
    [ $(( shm_gb * 80 / 100 )) -lt "$KV_OFFLOAD_GB" ] && KV_OFFLOAD_GB=$(( shm_gb * 80 / 100 ))
    [ "$KV_OFFLOAD_GB" -gt 96 ] && KV_OFFLOAD_GB=96
    echo "[p2] host MemAvailable ${avail_gb} GiB, /dev/shm free ${shm_gb} GiB -> KV offload ${KV_OFFLOAD_GB} GiB"
  fi
  [ "$KV_OFFLOAD_GB" -ge 4 ] 2>/dev/null && OFFLOAD_ARGS=(--kv-offloading-size "$KV_OFFLOAD_GB" --kv-offloading-backend native)
fi
MM_ARGS=()
if [ -n "${MM_LIMIT:-}" ]; then MM_ARGS=(--limit-mm-per-prompt "$MM_LIMIT"); echo "[p2] multimodal limit: $MM_LIMIT"; fi
THINK_ARGS=()
if [ "${THINK:-0}" = "1" ]; then
  if /root/vllm/bin/python -m vllm.entrypoints.openai.api_server --help 2>/dev/null | grep -q -- "--default-chat-template-kwargs"; then
    THINK_ARGS=(--default-chat-template-kwargs '{"enable_thinking": true}')
    echo "[p2] thinking: enable_thinking=true via --default-chat-template-kwargs"
  else
    echo "[p2] WARN: vLLM lacks --default-chat-template-kwargs; thinking left at the template default"
  fi
fi

start_server() {   # $1 = mode label, rest = extra vLLM flags; returns 0 when healthy
  local label="$1"; shift
  echo "[p2] starting vLLM ($label) at $(date -u +%H:%M:%S)"
  nohup /root/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
    --dtype bfloat16 --max-model-len "$MAXLEN_EFF" --gpu-memory-utilization "${GPU_UTIL:-0.92}" --seed 0 "$@" \
    --reasoning-parser "$RPARSER" --enable-auto-tool-choice --tool-call-parser "$TOOLPARSER" "${THINK_ARGS[@]}" "${MM_ARGS[@]}" "${OFFLOAD_ARGS[@]}" \
    >> "$ABS_OUT/vllm_server.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "[p2] server healthy after ${i}0s at $(date -u +%H:%M:%S) (mode: $label)"; SERVE_MODE="$label"; return 0; fi
    if ! kill -0 $SRV 2>/dev/null; then
      echo "[p2] server died (mode: $label); last errors:"
      grep -E "Error|error:|ValueError|RuntimeError|larger than|KV cache" "$ABS_OUT/vllm_server.log" | tail -6
      if [ ${#OFFLOAD_ARGS[@]} -gt 0 ] && tail -120 "$ABS_OUT/vllm_server.log" | grep -qiE "offload|kv.transfer|kv_connector|OffloadingConnector"; then
        echo "[p2] KV offloading rejected for this model; retrying $label WITHOUT offloading"; OFFLOAD_ARGS=(); sleep 5; start_server "$label" "$@"; return $?; fi
      if [ ${#MM_ARGS[@]} -gt 0 ] && tail -80 "$ABS_OUT/vllm_server.log" | grep -qiE "limit.mm.per.prompt|modalit"; then
        echo "[p2] multimodal limit rejected; retrying $label without it"; MM_ARGS=(); sleep 5; start_server "$label" "$@"; return $?; fi
      return 1; fi
    sleep 10
  done
  echo "[p2] server never became healthy (mode: $label)"; kill $SRV 2>/dev/null; return 1
}
T_SERVE0=$(date +%s)
# bf16 KV first (compiled, then eager); fp8 KV only if bf16 cannot hold one max-length request.
start_server "compiled/bf16-kv" \
  || { sleep 5; start_server "eager/bf16-kv" --enforce-eager; } \
  || { sleep 5; start_server "compiled/fp8-kv" --kv-cache-dtype fp8; } \
  || { sleep 5; start_server "eager/fp8-kv" --enforce-eager --kv-cache-dtype fp8; } \
  || { echo "[p2] FATAL server failed in every mode"; exit 21; }
echo "[p2] serve-ready seconds (download+load+warmup): $(( $(date +%s) - T_SERVE0 ))  mode=$SERVE_MODE"
echo "[p2] unauthenticated-HF warnings in server log: $(grep -c 'unauthenticated requests' "$ABS_OUT/vllm_server.log")"
echo "[p2] KV offloading: ${OFFLOAD_ARGS[*]:-off}; gpu_util=${GPU_UTIL:-0.92}"
grep -iE "OffloadingConnector|offloading" "$ABS_OUT/vllm_server.log" | sed -E 's/^.*\] /[p2] /' | head -3
grep -E "GPU KV cache size|Maximum concurrency|Model loading took" "$ABS_OUT/vllm_server.log" | sed -E 's/^.*\] /[p2] /' | tail -3

# 5b. Optional context-budget check (does vLLM reject prompt + max_tokens > max-model-len, as the harness needs?)
if [ "${CTX_CHECK:-0}" = "1" ]; then
  /root/vllm/bin/python "$POD/ctx_check.py" "http://localhost:$PORT" "$SERVED" "$MAXLEN_EFF" "$MAXTOK" || echo "[p2] WARN ctx_check failed"
fi

# 6. Harness venv
uv venv -q /root/ecenv --python 3.12 || { echo "[p2] FATAL ecenv create failed"; exit 23; }
VIRTUAL_ENV=/root/ecenv uv pip install -q -r "$ECDIR/requirements.txt" || { echo "[p2] FATAL ec deps install failed"; exit 24; }


# 7. Published peer cache (reps used by any seed)
REPS=$(cat "$SMOKE" $( [ "$MAINJ" != none ] && echo "$MAINJ") | cut -f1 | sed -E 's#.*_rep([0-9]{3})__.*#\1#' | sort -u | tr '\n' ' ')
echo "[p2] fetching published peer cache reps: $REPS"
/root/vllm/bin/python - "$PEER_ROOT" $REPS <<'PY' || { echo "[p2] FATAL peer cache download failed"; exit 27; }
import sys
from huggingface_hub import snapshot_download
root, reps = sys.argv[1], sys.argv[2:]
snapshot_download("SALT-NLP/agent-collusion", repo_type="dataset", local_dir=root,
                  allow_patterns=[f"peer-cache/rep{r}/*.json" for r in reps])
PY
for r in $REPS; do echo "[p2]   rep$r: $(ls "$PEER_ROOT/peer-cache/rep$r" 2>/dev/null | wc -l) files"; done

# 8. Verify every seed record once (restored == saved, both agents); unverified seeds are skipped
cd "$ECDIR" || exit 25
export OPENAI_API_BASE="http://localhost:$PORT/v1" OPENAI_API_KEY="sk-local-dummy"
declare -A VERIFIED
mkdir -p "$ABS_OUT/verify"
for rec in $(cat "$SMOKE" $( [ "$MAINJ" != none ] && echo "$MAINJ") | cut -f1 | sort -u); do
  vlog="$ABS_OUT/verify/$(echo "$rec" | tr '/' '_').log"
  if /root/ecenv/bin/python -I "$POD/verify_resume.py" "$ECDIR" "/root/NEEL/$rec" > "$vlog" 2>&1; then VERIFIED[$rec]=1; s=PASS; else VERIFIED[$rec]=0; s=FAIL; fi
  echo "[p2] verify $s $rec ($(grep -c 'identical=True' "$vlog")/2 agents identical)"
done

RPIDS=()
alive_runners() { local n=0 p; for p in "${RPIDS[@]}"; do kill -0 "$p" 2>/dev/null && n=$((n+1)); done; echo $n; }
heartbeat() {
  local met=$(grep -E "Avg prompt throughput" "$ABS_OUT/vllm_server.log" | tail -1 | sed -E 's/^.*Avg prompt/prompt/; s/ tokens\/s//g' | cut -c1-150)
  echo "[p2] $(date -u +%H:%M) $1 +$(( ($(date +%s)-$2)/60 ))m running=$(alive_runners) saved=$(grep -l 'Saved incremental results after' $(find "$ABS_OUT" -name runner.log) 2>/dev/null | wc -l) | $met"
}
run_phase() {   # $1 = jobs tsv, $2 = max resample index (exclusive), $3 = label
  local tsv="$1" rmax="$2" label="$3"
  local t0=$(date +%s) launched=0 lastbeat=$(date +%s)
  RPIDS=()
  while IFS=$'\t' read -r REC SUB R; do
    [ -z "${REC:-}" ] && continue
    lim="$rmax"; case "$SUB" in *__probe) lim="${PROBE_R:-$rmax}" ;; esac
    [ "$R" -ge "$lim" ] && continue
    [ "${VERIFIED[$REC]:-0}" = 1 ] || { echo "[p2] SKIP unverified $REC"; continue; }
    while [ "$(alive_runners)" -ge "${CONC:-3}" ]; do
      sleep 10
      if [ $(( $(date +%s) - lastbeat )) -ge "${HEARTBEAT_S:-300}" ]; then lastbeat=$(date +%s); heartbeat "$label" "$t0"; fi
    done
    OUT="$ABS_OUT/$SUB/r$(printf %02d "$R")"; mkdir -p "$OUT"
    mapfile -t ARGS < <(/root/ecenv/bin/python -I "$POD/replay_cmd.py" "/root/NEEL/$REC" "$OUT")
    /root/ecenv/bin/python -m experiments.runner "${ARGS[@]}" > "$OUT/runner.log" 2>&1 < /dev/null &
    RPIDS+=($!); launched=$((launched+1))
  done < "$tsv"
  echo "[p2] $label: $launched replay episodes launched (conc ${CONC:-3}); waiting"
  while [ "$(alive_runners)" -gt 0 ]; do
    sleep 10
    if [ $(( $(date +%s) - lastbeat )) -ge "${HEARTBEAT_S:-300}" ]; then lastbeat=$(date +%s); heartbeat "$label" "$t0"; fi
  done
  echo "[p2] $label finished in $(( ($(date +%s)-t0)/60 )) min; runner logs with a saved episode: $(grep -l 'Saved incremental results after' $(find "$ABS_OUT" -name runner.log) 2>/dev/null | wc -l); with a Traceback: $(grep -l Traceback $(find "$ABS_OUT" -name runner.log) 2>/dev/null | wc -l)"
}

# 9. Phase 1: smoke
run_phase "$SMOKE" 999 smoke
# 10. Phase 2: main, after the go file (content = resamples R)
if [ "$MAINJ" != none ]; then
  [ -n "${GO_R:-}" ] && echo "$GO_R" > /root/p2_go   # pre-approved R (no smoke on this pod)
  echo "[p2] waiting up to ${GO_TIMEOUT_MIN:-25} min for /root/p2_go (resamples R) before the main phase"
  for i in $(seq 1 $(( ${GO_TIMEOUT_MIN:-25} * 6 ))); do [ -f /root/p2_go ] && break; sleep 10; done
  if [ -f /root/p2_go ]; then
    read -r RMAX PROBE_R < /root/p2_go; PROBE_R=${PROBE_R:-$RMAX}; export PROBE_R
    echo "[p2] go received: R=$RMAX (probe R=$PROBE_R)"
    run_phase "$MAINJ" "$RMAX" main
  else
    echo "[p2] no go file within ${GO_TIMEOUT_MIN:-25} min; skipping main phase"
  fi
fi
kill $SRV 2>/dev/null
echo "[p2] all done"
exit 0
