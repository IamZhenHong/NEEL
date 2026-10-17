#!/usr/bin/env bash
# Exp 74 on one A100 80GB. Per model: vLLM server -> runner (generation) -> stop -> HF reads (project venv).
#   STAGE=pilot: units/pilot.jsonl; HF: extract (direction, layer, floors) + pilot unit reads
#   STAGE=main : units/main.jsonl;  HF: unit reads with the frozen directions (coherence/frozen/dirs_<model>.npz)
#   MODELS="qwen gemma" (default) | LIMIT=n (first n units, smoke)
# Launch (repo root):
#   caffeinate -i uv run python infra/launch_run.py --name e74-<stage> --allow-concurrent --max-minutes M --max-cost C \
#     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 200 --exclude ... \
#     --results-dir coherence/results/<stage> -- bash coherence/pod.sh   (with STAGE set via env in the command)
set -uo pipefail
cd "$(dirname "$0")/.."
E=coherence
STAGE=${STAGE:-pilot}
MODELS=${MODELS:-"qwen gemma"}
UNITS=${UNITS:-$STAGE}          # unit file name (actpilot uses pilot)
RARGS=${RARGS:-}               # extra runner args (actpilot: --only act --act-samples 3)
READS=${READS:-1}              # 0 = no HF stage
R=$E/results/$STAGE
mkdir -p $R/gen $R/reads $R/server
LOG=$R/pod.log
echo "[e74] stage $STAGE models $MODELS start $(date -u +%H:%M:%S) rev ${GIT_REV:-?}" | tee -a $LOG
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $LOG
df -h /root | tail -1 | tee -a $LOG
bash vigilante/pod/setup_vllm.sh 2>&1 | tail -4 | tee -a $LOG
export VLLM_USE_FLASHINFER_SAMPLER=0
VPY=/root/vllm/bin/python
SPID=""

stop() { [ -n "$SPID" ] && kill $SPID 2>/dev/null; sleep 5; [ -n "$SPID" ] && kill -9 $SPID 2>/dev/null; pkill -9 -f "vllm serve"; pkill -9 -f EngineCore; sleep 8; SPID=""; }
serve() {   # name model max_model_len
  local name=$1 model=$2 mml=$3 rev
  rev=$($VPY -c "from huggingface_hub import HfApi; print(HfApi().model_info('$model').sha)")
  echo "[e74] serving $name: $model@$rev mml $mml" | tee -a $LOG
  for mm in '{"image": 0, "video": 0}' '{"image": 0}' ''; do
    /root/vllm/bin/vllm serve "$model" --revision "$rev" --served-model-name m --port 8000 --dtype bfloat16 \
        --max-model-len "$mml" --gpu-memory-utilization 0.92 --seed 0 --enable-prefix-caching \
        ${mm:+--limit-mm-per-prompt "$mm"} > $R/server/${name}_$(date +%H%M%S).log 2>&1 &
    SPID=$!
    for i in $(seq 1 150); do
      if curl -sf http://127.0.0.1:8000/health > /dev/null; then
        echo "[e74] $name up after $((i*10))s (mm limit: ${mm:-none})" | tee -a $LOG
        echo "{\"model\": \"$model\", \"revision\": \"$rev\", \"max_model_len\": $mml}" > $R/server/${name}.json
        return 0
      fi
      if ! kill -0 $SPID 2>/dev/null; then
        echo "[e74] $name server died (mm limit: ${mm:-none})" | tee -a $LOG
        tail -25 $(ls -t $R/server/${name}_*.log | head -1) | tee -a $LOG
        break
      fi
      sleep 10
    done
    stop
  done
  echo "[e74] $name could not be served" | tee -a $LOG
  return 1
}

for M in $MODELS; do
  case $M in
    qwen)  HF=Qwen/Qwen3.6-27B;      FMT=qwen;   MML=16384; W=64 ;;
    gemma) HF=google/gemma-4-31B-it; FMT=gemma4; MML=12288; W=32 ;;
  esac
  if serve $M $HF $MML; then
    $VPY -m coherence.runner --units $E/units/$UNITS.jsonl --out $R/gen/$M --fmt $FMT --model $HF \
        --server http://127.0.0.1:8000 --workers $W ${LIMIT:+--limit $LIMIT} $RARGS 2>&1 | tee -a $LOG
    grep -E "throughput|Prefix cache hit" $(ls -t $R/server/${M}_*.log | head -1) | tail -2 | tee -a $LOG
  fi
  stop
  [ "$READS" = 0 ] && { echo "[e74] $M done (no reads) $(date -u +%H:%M:%S)" | tee -a $LOG; continue; }
  if [ "$STAGE" = pilot ]; then
    python -m coherence.pod_read extract --model $M --out $R/reads/$M 2>&1 \
      | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a $LOG
    DIRS=$R/reads/$M/dirs.npz
  else
    DIRS=$E/frozen/dirs_$M.npz
  fi
  python -m coherence.pod_read units --model $M --dirs $DIRS --units $E/units/$UNITS.jsonl --gen $R/gen/$M \
      --out $R/reads/$M 2>&1 | grep --line-buffered -v "Loading weights\|Fetching\|it/s\]" | tee -a $LOG
  echo "[e74] $M done $(date -u +%H:%M:%S)" | tee -a $LOG
done
du -sh $R | tee -a $LOG
echo "[e74] all done $(date -u +%H:%M:%S)" | tee -a $LOG
