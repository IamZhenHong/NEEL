#!/usr/bin/env bash
# s1 on one A100 80GB (launched by infra/launch_run.py; results dir selfknow/s1_persistence/results).
# 1. vLLM venv (vigilante/pod/setup_vllm.sh); Gemma weights prefetched in the background while Qwen runs.
# 2. Pilot (1k, variants 0-1): Qwen3.6-27B, then Gemma 4 31B, each its own process.
# 3. Wait for a control file pushed by scp into selfknow/s1_persistence/control/:
#      main  -> run the main stage (after the freeze commit/tag; code re-synced by scp first)
#      pilot -> re-run the pilot (probe wording fixed on the 1k context); the old pilot is kept as pilot_prev<N>
#      stop  -> exit (gate failed)
#    No control file within WAIT_MIN minutes -> exit.
# Engine settings per model are tried in order; pod_s1 exits 3 when vLLM cannot be built with a setting.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=selfknow/s1_persistence
R=$E/results
C=$E/control
mkdir -p $R $C
log() { echo "[s1pod $(date +%H:%M:%S)] $*" | tee -a $R/pod.log; }
log "start; $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
# heartbeat: the launcher aborts a job whose log is silent for 30 min (model output goes to files)
( while true; do sleep 240; echo "[s1pod hb $(date +%H:%M)] $(for f in $R/*/branches_*.jsonl; do [ -f $f ] && echo -n "$f:$(grep -c . $f) "; done) $(tail -c 300 $R/*/run_*.log 2>/dev/null | grep -o '\[runner\][^\n]*' | tail -1)"; done ) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT
bash vigilante/pod/setup_vllm.sh > $R/setup.log 2>&1 || { log "vllm setup FAILED"; tail -30 $R/setup.log; exit 1; }
log "vllm ready: $(grep '\[pod\] vllm' $R/setup.log)"
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
/root/vllm/bin/python -c "from huggingface_hub import snapshot_download as s; s('google/gemma-4-31B-it'); print('gemma prefetched')" \
    > $R/prefetch_gemma.log 2>&1 &

run_model() {   # $1 model key, $2 stage
  local mk=$1 st=$2 rc=1 cfg
  mkdir -p $R/$st
  if [ "$mk" = qwen ]; then
    CFGS=("--mml 32768 --gpu-mem 0.92" "--mml 28672 --gpu-mem 0.94")
  else
    CFGS=("--mml 32768 --mnbt 2048 --gpu-mem 0.93" "--mml 28672 --mnbt 2048 --gpu-mem 0.95"
          "--mml 28672 --mnbt 2048 --gpu-mem 0.95 --kv-dtype fp8")
  fi
  for cfg in "${CFGS[@]}"; do
    log "$mk $st: $cfg"
    /root/vllm/bin/python -m selfknow.s1_persistence.pod_s1 --model-key $mk --stage $st --out $R/$st $cfg \
        >> $R/$st/run_$mk.log 2>&1
    rc=$?
    log "$mk $st exit $rc ($(grep -c . $R/$st/branches_$mk.jsonl 2>/dev/null || echo 0) branches)"
    [ $rc -ne 3 ] && break
  done
  return $rc
}
stage() { run_model qwen $1; run_model gemma4 $1; touch $R/$1/DONE; log "stage $1 done"; }

stage pilot
WAIT=${WAIT_MIN:-25}
t0=$(date +%s)
n=0
while true; do
  if [ -f $C/main ]; then rm -f $C/main; log "control: main"; stage main; break; fi
  if [ -f $C/pilot ]; then
    rm -f $C/pilot; n=$((n + 1)); mv $R/pilot $R/pilot_prev$n; log "control: re-pilot ($n)"; stage pilot; t0=$(date +%s)
  fi
  if [ -f $C/stop ]; then log "control: stop"; break; fi
  if [ $(( $(date +%s) - t0 )) -gt $(( WAIT * 60 )) ]; then log "no control file in $WAIT min; exiting"; break; fi
  sleep 10
done
log "pod done"
