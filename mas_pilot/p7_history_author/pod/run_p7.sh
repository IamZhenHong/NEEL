#!/usr/bin/env bash
# P7 driver (Qwen3.6-27B, one pod). Serving, harness venv, peer caches, variant()/run_jobs() as run_p345.sh (copied
# verbatim up to its job section). Then: verify every P7 variant record (arm a1 cache) -> mechanics smoke (one
# trajectory x 7 arms x 1 resample; gate: all verified, >= 6/7 valid verdicts) -> main run (7 arms x 8 trajectories x
# R resamples, resample-major; P/O version i in resample i) -> off-policy distance (prompt log-probs) on the live server.
# Args: MODE(=P7) OUTROOT   Env: R (default 3), CONC (default 8), MAXLEN THINK GPU_UTIL KV_OFFLOAD_GB MM_LIMIT HEARTBEAT_S
set -uo pipefail
MODE="$1"; OUTROOT="$2"
MODEL=Qwen/Qwen3.6-27B; SERVED=qwen3.6-27b; MAXTOK=14848; RPARSER=qwen3; TOOLPARSER=hermes
PORT=8000
POD=/root/NEEL/mas_pilot/p3_p4_p5/pod
SEEDS=/root/NEEL/mas_pilot/p3_p4_p5/seeds
ECDIR=/root/NEEL/mas_pilot/p1_baserate/ext/ec_repo
ABS_OUT=/root/NEEL/$OUTROOT
PEER_ROOT=/root/peer_dl
CACHE_ROOT=/root/p345_cache
RR=${R:-3}
mkdir -p "$ABS_OUT"
export PATH="$HOME/.local/bin:$PATH"
echo "[p345] mode=$MODE out=$OUTROOT R=$RR conc=${CONC:-8} maxlen=${MAXLEN:-131072} think=${THINK:-0} gpu_util=${GPU_UTIL:-0.92}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. vLLM venv (+ninja)
bash "/root/NEEL/mas_pilot/p2_replay/pod/setup_vllm.sh" || { echo "[p345] FATAL vllm setup failed"; exit 20; }

# 2. HF token (never printed), ninja on PATH, native sampler
if [ -z "${HF_TOKEN:-}" ]; then HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^HF_TOKEN=//p'); export HF_TOKEN; fi
[ -n "${HF_TOKEN:-}" ] && echo "[p345] HF_TOKEN present (len ${#HF_TOKEN})" || echo "[p345] WARN: HF_TOKEN absent (gated models will fail)"
(apt-get install -y -qq ninja-build >/dev/null 2>&1 || (apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq ninja-build >/dev/null 2>&1)) || true
export PATH="/root/vllm/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
command -v ninja >/dev/null && echo "[p345] ninja $(ninja --version)" || echo "[p345] WARN: ninja not on PATH"

# 3. Parsers (auto = resolve from the installed vLLM source)
if [ "$RPARSER" = "auto" ] || [ "$TOOLPARSER" = "auto" ]; then
  read -r RP_AUTO TP_AUTO <<< "$(/root/vllm/bin/python "/root/NEEL/mas_pilot/p2_replay/pod/resolve_parsers.py" "$MODEL")"
  [ "$RPARSER" = "auto" ] && RPARSER="$RP_AUTO"
  [ "$TOOLPARSER" = "auto" ] && TOOLPARSER="$TP_AUTO"
fi
echo "[p345] parsers: reasoning=$RPARSER tool=$TOOLPARSER"
if [ "$RPARSER" = "NONE" ] || [ "$TOOLPARSER" = "NONE" ] || [ -z "$RPARSER" ] || [ -z "$TOOLPARSER" ]; then
  echo "[p345] FATAL no matching parser for $MODEL"; exit 26; fi

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
echo "[p345] model max_position_embeddings=${MAXPOS:-unknown}; serving max-model-len=$MAXLEN_EFF"

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
    echo "[p345] host MemAvailable ${avail_gb} GiB, /dev/shm free ${shm_gb} GiB -> KV offload ${KV_OFFLOAD_GB} GiB"
  fi
  [ "$KV_OFFLOAD_GB" -ge 4 ] 2>/dev/null && OFFLOAD_ARGS=(--kv-offloading-size "$KV_OFFLOAD_GB" --kv-offloading-backend native)
fi
MM_ARGS=()
if [ -n "${MM_LIMIT:-}" ]; then MM_ARGS=(--limit-mm-per-prompt "$MM_LIMIT"); echo "[p345] multimodal limit: $MM_LIMIT"; fi
THINK_ARGS=()
if [ "${THINK:-0}" = "1" ]; then
  if /root/vllm/bin/python -m vllm.entrypoints.openai.api_server --help 2>/dev/null | grep -q -- "--default-chat-template-kwargs"; then
    THINK_ARGS=(--default-chat-template-kwargs '{"enable_thinking": true}')
    echo "[p345] thinking: enable_thinking=true via --default-chat-template-kwargs"
  else
    echo "[p345] WARN: vLLM lacks --default-chat-template-kwargs; thinking left at the template default"
  fi
fi

start_server() {   # $1 = mode label, rest = extra vLLM flags; returns 0 when healthy
  local label="$1"; shift
  echo "[p345] starting vLLM ($label) at $(date -u +%H:%M:%S)"
  nohup /root/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
    --dtype bfloat16 --max-model-len "$MAXLEN_EFF" --gpu-memory-utilization "${GPU_UTIL:-0.92}" --seed 0 "$@" \
    --reasoning-parser "$RPARSER" --enable-auto-tool-choice --tool-call-parser "$TOOLPARSER" "${THINK_ARGS[@]}" "${MM_ARGS[@]}" "${OFFLOAD_ARGS[@]}" \
    >> "$ABS_OUT/vllm_server.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then
      echo "[p345] server healthy after ${i}0s at $(date -u +%H:%M:%S) (mode: $label)"; SERVE_MODE="$label"; return 0; fi
    if ! kill -0 $SRV 2>/dev/null; then
      echo "[p345] server died (mode: $label); last errors:"
      grep -E "Error|error:|ValueError|RuntimeError|larger than|KV cache" "$ABS_OUT/vllm_server.log" | tail -6
      if [ ${#OFFLOAD_ARGS[@]} -gt 0 ] && tail -120 "$ABS_OUT/vllm_server.log" | grep -qiE "offload|kv.transfer|kv_connector|OffloadingConnector"; then
        echo "[p345] KV offloading rejected for this model; retrying $label WITHOUT offloading"; OFFLOAD_ARGS=(); sleep 5; start_server "$label" "$@"; return $?; fi
      if [ ${#MM_ARGS[@]} -gt 0 ] && tail -80 "$ABS_OUT/vllm_server.log" | grep -qiE "limit.mm.per.prompt|modalit"; then
        echo "[p345] multimodal limit rejected; retrying $label without it"; MM_ARGS=(); sleep 5; start_server "$label" "$@"; return $?; fi
      return 1; fi
    sleep 10
  done
  echo "[p345] server never became healthy (mode: $label)"; kill $SRV 2>/dev/null; return 1
}
T_SERVE0=$(date +%s)
# bf16 KV first (compiled, then eager); fp8 KV only if bf16 cannot hold one max-length request.
start_server "compiled/bf16-kv" \
  || { sleep 5; start_server "eager/bf16-kv" --enforce-eager; } \
  || { sleep 5; start_server "compiled/fp8-kv" --kv-cache-dtype fp8; } \
  || { sleep 5; start_server "eager/fp8-kv" --enforce-eager --kv-cache-dtype fp8; } \
  || { echo "[p345] FATAL server failed in every mode"; exit 21; }
echo "[p345] serve-ready seconds (download+load+warmup): $(( $(date +%s) - T_SERVE0 ))  mode=$SERVE_MODE"
echo "[p345] unauthenticated-HF warnings in server log: $(grep -c 'unauthenticated requests' "$ABS_OUT/vllm_server.log")"
echo "[p345] KV offloading: ${OFFLOAD_ARGS[*]:-off}; gpu_util=${GPU_UTIL:-0.92}"
grep -iE "OffloadingConnector|offloading" "$ABS_OUT/vllm_server.log" | sed -E 's/^.*\] /[p345] /' | head -3
grep -E "GPU KV cache size|Maximum concurrency|Model loading took" "$ABS_OUT/vllm_server.log" | sed -E 's/^.*\] /[p345] /' | tail -3

# 5b. Optional context-budget check (does vLLM reject prompt + max_tokens > max-model-len, as the harness needs?)
if [ "${CTX_CHECK:-0}" = "1" ]; then
  /root/vllm/bin/python "$POD/ctx_check.py" "http://localhost:$PORT" "$SERVED" "$MAXLEN_EFF" "$MAXTOK" || echo "[p345] WARN ctx_check failed"
fi

# 6. Harness venv
uv venv -q /root/ecenv --python 3.12 || { echo "[p345] FATAL ecenv create failed"; exit 23; }
VIRTUAL_ENV=/root/ecenv uv pip install -q -r "$ECDIR/requirements.txt" || { echo "[p345] FATAL ec deps install failed"; exit 24; }



# 7. Published peer cache (reps 001-003) and per-arm caches with the extra round-5 message
/root/vllm/bin/python - "$PEER_ROOT" <<'PY' || { echo "[p345] FATAL peer cache download failed"; exit 27; }
import sys
from huggingface_hub import snapshot_download
snapshot_download("SALT-NLP/agent-collusion", repo_type="dataset", local_dir=sys.argv[1],
                  allow_patterns=[f"peer-cache/rep00{r}/*.json" for r in (1, 2, 3)])
PY
/root/ecenv/bin/python -I "$POD/prep_caches.py" "$PEER_ROOT" "$CACHE_ROOT" 001 002 003 || { echo "[p345] FATAL caches"; exit 28; }
STANCE=$(/root/ecenv/bin/python -I -c "import sys; sys.path.insert(0,'$POD'); from p345_common import STANCE; print(STANCE)")

cd "$ECDIR" || exit 25
export OPENAI_API_BASE="http://localhost:$PORT/v1" OPENAI_API_KEY="sk-local-dummy"
VDIR="$ABS_OUT/variants"; mkdir -p "$VDIR" "$ABS_OUT/verify"
declare -A VERIFIED

variant() {   # $1 base record, $2 name, $3 arm, $4 stance(0/1), $5 compact json or ""
  local out="$VDIR/$2.json" args=(--arm "$3")
  [ "$4" = 1 ] && args+=(--stance)
  [ -n "$5" ] && args+=(--compact "$5")
  /root/ecenv/bin/python -I "$POD/make_variants.py" "$1" "$out" "$CACHE_ROOT" "${args[@]}" || return 1
  local vlog="$ABS_OUT/verify/$2.log"
  if [ "$4" = 1 ]; then P345_ALICE_STANCE="$STANCE" /root/ecenv/bin/python -I "$POD/verify_p345.py" "$ECDIR" "$out" > "$vlog" 2>&1
  else /root/ecenv/bin/python -I "$POD/verify_p345.py" "$ECDIR" "$out" > "$vlog" 2>&1; fi
  if [ $? -eq 0 ]; then VERIFIED[$2]=1; else VERIFIED[$2]=0; echo "[p345] verify FAIL $2"; fi
}

RPIDS=()
alive() { local n=0 p; for p in "${RPIDS[@]}"; do kill -0 "$p" 2>/dev/null && n=$((n+1)); done; echo $n; }
heartbeat() {
  local met=$(grep -E "Avg prompt throughput" "$ABS_OUT/vllm_server.log" | tail -1 | sed -E 's/^.*Avg prompt/prompt/; s/ tokens\/s//g' | cut -c1-140)
  echo "[p345] $(date -u +%H:%M) $1 +$(( ($(date +%s)-$2)/60 ))m running=$(alive) saved=$(grep -l 'Saved incremental results after' $(find "$ABS_OUT" -name runner.log) 2>/dev/null | wc -l) | $met"
}
run_jobs() {   # $1 jobs tsv (variant_name TAB subdir TAB r TAB stance), $2 label
  local t0=$(date +%s) lastbeat=$(date +%s) n=0
  RPIDS=()
  while IFS=$'\t' read -r V SUB R ST; do
    [ -z "${V:-}" ] && continue
    [ "${VERIFIED[$V]:-0}" = 1 ] || { echo "[p345] SKIP unverified $V"; continue; }
    while [ "$(alive)" -ge "${CONC:-8}" ]; do
      sleep 10
      if [ $(( $(date +%s) - lastbeat )) -ge "${HEARTBEAT_S:-300}" ]; then lastbeat=$(date +%s); heartbeat "$2" "$t0"; fi
    done
    local OUT="$ABS_OUT/$SUB/r$(printf %02d "$R")"; mkdir -p "$OUT"
    mapfile -t ARGS < <(/root/ecenv/bin/python -I "$POD/replay_cmd_p345.py" "$VDIR/$V.json" "$OUT")
    if [ "$ST" = 1 ]; then
      P345_ALICE_STANCE="$STANCE" /root/ecenv/bin/python "$POD/p345_runner.py" "$ECDIR" "${ARGS[@]}" > "$OUT/runner.log" 2>&1 < /dev/null &
    else
      /root/ecenv/bin/python "$POD/p345_runner.py" "$ECDIR" "${ARGS[@]}" > "$OUT/runner.log" 2>&1 < /dev/null &
    fi
    RPIDS+=($!); n=$((n+1))
  done < "$1"
  echo "[p345] $2: $n replays launched; waiting"
  while [ "$(alive)" -gt 0 ]; do
    sleep 10
    if [ $(( $(date +%s) - lastbeat )) -ge "${HEARTBEAT_S:-300}" ]; then lastbeat=$(date +%s); heartbeat "$2" "$t0"; fi
  done
  echo "[p345] $2 finished in $(( ($(date +%s)-t0)/60 )) min"
}
accept_at() {   # $1 = replay subdir root -> prints "accepted total"
  /root/ecenv/bin/python -I - "$1" <<'PY'
import glob, json, sys
k = n = 0
for p in glob.glob(sys.argv[1] + "/**/run.json", recursive=True):
    res = json.load(open(p))["results"]
    if res:
        n += 1
        k += str(res[-1]["analysis_metadata"]["verdict_by_agent"]["alice"]).lower() == "accept"
print(k, n)
PY
}

P7=/root/NEEL/mas_pilot/p7_history_author
P7SEEDS=$P7/seeds
SMOKE_BASE=qwen_compliant_rep001__E5
NV=0; NF=0
while IFS=$'\t' read -r NAME BASE TRAJ ARM VER; do
  [ "$NAME" = name ] && continue
  variant "$P7SEEDS/$NAME/record.json" "$NAME" a1 0 ""
  if [ "${VERIFIED[$NAME]}" = 1 ]; then NV=$((NV+1)); else NF=$((NF+1)); fi
done < "$P7SEEDS/p7_seeds.tsv"
echo "[p7] variants verified: $NV, failed: $NF"

valid_at() {   # $1 = replay subdir root -> prints "valid total accepted"
  /root/ecenv/bin/python -I - "$1" <<'PY'
import glob, json, sys
v = n = a = 0
for p in glob.glob(sys.argv[1] + "/**/run.json", recursive=True):
    res = json.load(open(p))["results"]
    n += 1
    if res:
        x = str(res[-1]["analysis_metadata"]["verdict_by_agent"]["alice"]).lower()
        v += x in ("accept", "reject"); a += x == "accept"
print(v, n, a)
PY
}

# ---- mechanics smoke
: > "$ABS_OUT/jobs_smoke.tsv"
SMOKE_FAIL=0
for ARM in U P O S N U-x S-x; do
  V="${SMOKE_BASE}__${ARM}__v0"
  [ "${VERIFIED[$V]:-0}" = 1 ] || { echo "[p7] smoke: $V not verified"; SMOKE_FAIL=1; }
  printf '%s\t%s\t%s\t0\n' "$V" "smoke/$ARM" 0 >> "$ABS_OUT/jobs_smoke.tsv"
done
run_jobs "$ABS_OUT/jobs_smoke.tsv" smoke
read -r SV SN SA <<< "$(valid_at "$ABS_OUT/smoke")"
echo "[p7] SMOKE: valid verdicts $SV/$SN (accept $SA); all verified: $([ $SMOKE_FAIL = 0 ] && echo yes || echo no)"
if [ "$SMOKE_FAIL" != 0 ] || [ "${SV:-0}" -lt 6 ]; then echo "[p7] SMOKE GATE FAILED; stopping"; kill $SRV 2>/dev/null; exit 30; fi
echo "[p7] smoke gate passed"

# ---- main run, resample-major
: > "$ABS_OUT/jobs_main.tsv"
for r in $(seq 0 $((RR-1))); do
  while IFS=$'\t' read -r NAME BASE TRAJ ARM VER; do
    [ "$NAME" = name ] && continue
    case "$ARM" in P|O) want=$r ;; *) want=0 ;; esac
    [ "$VER" = "$want" ] || continue
    printf '%s\t%s\t%s\t0\n' "$NAME" "p7/$ARM/$TRAJ" "$r" >> "$ABS_OUT/jobs_main.tsv"
  done < "$P7SEEDS/p7_seeds.tsv"
done
echo "[p7] main jobs: $(wc -l < "$ABS_OUT/jobs_main.tsv")"
run_jobs "$ABS_OUT/jobs_main.tsv" main
for ARM in U P O S N U-x S-x; do echo "[p7] $ARM: valid/total/accept $(valid_at "$ABS_OUT/p7/$ARM")"; done

# ---- off-policy distance (descriptive)
/root/vllm/bin/python -I "$P7/pod/p7_logprob.py" "http://localhost:$PORT" "$SERVED" "$MODEL" "$P7SEEDS/texts.jsonl" \
  "$ABS_OUT/distance.jsonl" || echo "[p7] WARN distance step failed"
kill $SRV 2>/dev/null
echo "[p345] all done"
exit 0
