#!/usr/bin/env bash
# I1 pod job (started by the start command of mas_pilot/p3_p4_p5/pod/p5_http_launch.py once the data file is in
# /root/in; code in /root/code; only flat files in /root/out are served back).
# Serving exactly as P1/P4 Qwen (run_p345.sh): Qwen/Qwen3.6-27B on vLLM 0.30.0 (setup_vllm.sh), bf16, seed 0,
# qwen3 reasoning parser, hermes tool parser, thinking on, max-model-len 81920, gpu util 0.94, CPU KV offload sized
# to /dev/shm, no image/video. EC harness at commit 5afb2c5 (cloned here, on the pod only), published peer cache
# rep001-003 + the a1 extra round-5 message (prep_caches.py), variants via make_variants.py, every variant verified
# with verify_p345.py before it may run; replays through p345_runner.py, orchestrated by i1_orch.py.
set -uo pipefail
T_JOB0=$(date +%s); export T_JOB0
CODE=/root/code; IN=/root/in; OUT=/root/out; W=/root/work
mkdir -p "$OUT" "$W" /root/NEEL
export PATH="$HOME/.local/bin:$PATH" HF_HOME=/root/hf
MODEL=Qwen/Qwen3.6-27B; SERVED=qwen3.6-27b; PORT=8000
ECDIR=/root/ext/ec_repo; CACHE_ROOT=/root/p345_cache; PEER_ROOT=/root/peer_dl
log() { echo "[i1 +$(( ($(date +%s)-T_JOB0)/60 ))m] $*"; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
tar -xzf "$IN/i1_inputs.tar.gz" -C /root/NEEL || { log "FATAL inputs"; exit 30; }
log "inputs: $(ls /root/NEEL/i1_variants | wc -l) variant dirs"

# 1. uv, EC repo at the P1 commit (pod only)
command -v uv >/dev/null || curl -LsSf --max-time 300 https://astral.sh/uv/install.sh | sh >/dev/null
mkdir -p "$ECDIR" && cd "$ECDIR" && git init -q && git remote add origin https://github.com/SALT-NLP/agent-collusion
GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 origin 5afb2c595aa3eff6b9b829a032f4f117b36ee54d && git checkout -q FETCH_HEAD \
  || { log "FATAL ec clone"; exit 31; }
log "EC repo at $(git rev-parse HEAD)"
cd /root

# 2. vLLM venv; weights download in the background meanwhile
( /usr/bin/env python3 -m pip install -q huggingface_hub >/dev/null 2>&1; python3 -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL', max_workers=16)" >/root/dl.log 2>&1; echo "dl exit $?" >> /root/dl.log ) &
DLPID=$!
bash "$CODE/setup_vllm.sh" || { log "FATAL vllm setup"; exit 20; }
(apt-get install -y -qq ninja-build >/dev/null 2>&1 || (apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq ninja-build >/dev/null 2>&1)) || true
export PATH="/root/vllm/bin:$PATH" VLLM_USE_FLASHINFER_SAMPLER=0
log "vllm venv ready; waiting for weights"
wait $DLPID; tail -1 /root/dl.log

# 3. serve (flags as run_p345.sh with P1's Qwen env: MAXLEN=81920 THINK=1 GPU_UTIL=0.94 KV_OFFLOAD_GB=auto MM_LIMIT)
avail_gb=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
shm_gb=$(df -m /dev/shm | awk 'NR==2{printf "%d", $4/1024}')
KV=$(( avail_gb * 55 / 100 )); [ $(( shm_gb * 80 / 100 )) -lt "$KV" ] && KV=$(( shm_gb * 80 / 100 )); [ "$KV" -gt 96 ] && KV=96
OFFLOAD_ARGS=(); [ "$KV" -ge 4 ] && OFFLOAD_ARGS=(--kv-offloading-size "$KV" --kv-offloading-backend native)
log "RAM avail ${avail_gb} GiB, shm free ${shm_gb} GiB -> KV offload ${KV} GiB"
start_server() {
  local label="$1"; shift
  log "starting vLLM ($label)"
  nohup /root/vllm/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
    --dtype bfloat16 --max-model-len 81920 --gpu-memory-utilization 0.94 --seed 0 "$@" \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser hermes \
    --default-chat-template-kwargs '{"enable_thinking": true}' --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    "${OFFLOAD_ARGS[@]}" >> "$W/vllm_server.log" 2>&1 &
  SRV=$!
  for i in $(seq 1 180); do
    curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { log "server healthy ($label)"; return 0; }
    if ! kill -0 $SRV 2>/dev/null; then
      log "server died ($label):"; grep -E "Error|error:" "$W/vllm_server.log" | tail -5
      if [ ${#OFFLOAD_ARGS[@]} -gt 0 ] && tail -120 "$W/vllm_server.log" | grep -qiE "offload|kv.transfer|kv_connector"; then
        log "retrying without KV offload"; OFFLOAD_ARGS=(); sleep 5; start_server "$label" "$@"; return $?; fi
      return 1; fi
    sleep 10
  done
  kill $SRV 2>/dev/null; return 1
}
start_server compiled \
  || { sleep 5; start_server eager --enforce-eager; } \
  || { cp "$W/vllm_server.log" "$OUT/vllm_server.log"; log "FATAL server"; exit 21; }
grep -E "GPU KV cache size|Maximum concurrency|Model loading took" "$W/vllm_server.log" | sed -E 's/^.*\] //' | tail -3

# 4. harness venv, peer caches, variants + verification
uv venv -q /root/ecenv --python 3.12 && VIRTUAL_ENV=/root/ecenv uv pip install -q -r "$ECDIR/requirements.txt" \
  || { log "FATAL ecenv"; exit 24; }
/root/ecenv/bin/python - "$PEER_ROOT" <<'PY' || { log "FATAL peer cache"; exit 27; }
import sys
from huggingface_hub import snapshot_download
snapshot_download("SALT-NLP/agent-collusion", repo_type="dataset", local_dir=sys.argv[1],
                  allow_patterns=[f"peer-cache/rep00{r}/*.json" for r in (1, 2, 3)])
PY
/root/ecenv/bin/python -I "$CODE/prep_caches.py" "$PEER_ROOT" "$CACHE_ROOT" 001 002 003 | head -2
VDIR="$W/variants"; mkdir -p "$VDIR" "$W/verify"; : > "$W/jobs.tsv"; : > "$W/verify.tsv"
for r in 0 1 2; do : > "$W/jobs_r$r.tsv"; done
while IFS=$'\t' read -r NAME TRAJ SARM REP FLIP E STAGE REC; do
  [ "$STAGE" = p4k1 ] || continue
  for ARM in A B NEUT LAST NOFB NOREF; do
    for r in 0 1 2; do
      case $ARM in A|LAST|NOFB|NOREF) V="${NAME}__$ARM"; CF="/root/NEEL/i1_variants/$NAME/$ARM.json";;
                   *) V="${NAME}__$ARM$r"; CF="/root/NEEL/i1_variants/$NAME/$ARM$r.json";; esac
      if [ ! -f "$VDIR/$V.json" ]; then
        if [ $ARM = A ]; then /root/ecenv/bin/python -I "$CODE/make_variants.py" "/root/NEEL/$REC" "$VDIR/$V.json" "$CACHE_ROOT" --arm a1
        else /root/ecenv/bin/python -I "$CODE/make_variants.py" "/root/NEEL/$REC" "$VDIR/$V.json" "$CACHE_ROOT" --arm a1 --compact "$CF"; fi
        /root/ecenv/bin/python -I "$CODE/verify_p345.py" "$ECDIR" "$VDIR/$V.json" > "$W/verify/$V.log" 2>&1
        printf '%s\t%s\n' "$V" "$([ $? -eq 0 ] && echo PASS || echo FAIL)" >> "$W/verify.tsv"
      fi
      printf '%s\t%s\t%s\t%s\n' "$V" "$TRAJ" "$ARM" "$r" >> "$W/jobs_r$r.tsv"
    done
  done
done < /root/NEEL/mas_pilot/p3_p4_p5/seeds/base_seeds.tsv
log "verify: $(grep -c PASS "$W/verify.tsv") PASS, $(grep -c FAIL "$W/verify.tsv") FAIL"
(cd "$W" && tar -czf /root/verify.tar.gz verify verify.tsv) && mv /root/verify.tar.gz "$OUT/verify.tar.gz"
# resample-major; within each resample the 8 A replays go first (r0: anchor check), then the other arms round-robin
for r in 0 1 2; do
  /root/ecenv/bin/python -I - "$W/jobs_r$r.tsv" >> "$W/jobs.tsv" <<'PY'
import itertools, sys
rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1]) if l.strip()]
rest = {}
for x in rows:
    if x[2] != "A":
        rest.setdefault(x[2], []).append(x)
order = [x for x in rows if x[2] == "A"] + [x for grp in itertools.zip_longest(*rest.values()) for x in grp if x]
print("\n".join("\t".join(x) for x in order))
PY
done
log "jobs: $(wc -l < "$W/jobs.tsv")"

# 5. replays
cd "$ECDIR" || exit 25
export OPENAI_API_BASE="http://localhost:$PORT/v1" OPENAI_API_KEY="sk-local-dummy"
/root/ecenv/bin/python -I "$CODE/i1_orch.py" "$W/jobs.tsv" "$VDIR" "$W/verify.tsv" "$W/replays" "$OUT" "$CODE" "$ECDIR" "$W/vllm_server.log"
RC=$?
grep -E "GPU KV cache size|Maximum concurrency|Avg prompt throughput" "$W/vllm_server.log" | tail -4 | sed -E 's/^.*\] //' > /root/srv.txt; mv /root/srv.txt "$OUT/server_summary.txt"
kill $SRV 2>/dev/null
log "all done (orch exit $RC)"
exit $RC
