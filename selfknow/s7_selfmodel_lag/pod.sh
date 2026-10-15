#!/usr/bin/env bash
# s7 pod job (one A100 80GB). MODE=pilot: seed 1, 0% and 10% only (timing + self-model reads). MODE=main: seeds 1 and
# 2, 0..100% in 10% steps. Adapters go to adapters_pod/$MODE (outside the results dir); the local puller
# (pull_adapters.py) copies them during the run and marks each one pulled; at the end this script waits for it, then
# hands any adapter it did not get to the launcher's results pull.
#   MODE=main uv run python infra/launch_run.py --name s7-1 --allow-concurrent --max-minutes 150 --max-cost 4.5 \
#     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" "NVIDIA H100 80GB HBM3" --container-gb 160 \
#     --exclude story_imprinting/data/source --exclude story_imprinting/train/adapters \
#     --results-dir selfknow/s7_selfmodel_lag/results/main -- env MODE=main bash selfknow/s7_selfmodel_lag/pod.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
S7=selfknow/s7_selfmodel_lag
MODE=${MODE:?set MODE=pilot or main}
OUT=$S7/results/$MODE; ADIR=$S7/adapters_pod/$MODE
mkdir -p $OUT $ADIR
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $OUT/run.log
$PY -c "import torch, transformers, peft; print('[versions] torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee -a $OUT/run.log

# smoke on the pod (tiny model, ~1 min): stops the job before the paid part if anything is broken
$PY $S7/s7_run.py --model Qwen/Qwen3-0.6B --seed 1 --s2-n 32 --heldout-n 4 --eval-n 2 --samples 1 --selfdesc-n 2 \
    --max-new-tokens 8 --micro-batch 4 --gen-batch 8 --max-len 256 --stop-after 0.2 \
    --out /tmp/s7_smoke --adapter-dir /tmp/s7_smoke_ad > /tmp/s7_smoke.log 2>&1 \
    || { tail -30 /tmp/s7_smoke.log; echo "[smoke] FAILED"; exit 1; }
test -s /tmp/s7_smoke/selfmodel/s1_C_10.json && test -s /tmp/s7_smoke/replies/s1_C_0.jsonl \
    && test -s /tmp/s7_smoke_ad/s1_C_10/adapter_model.safetensors || { echo "[smoke] missing outputs"; ls -R /tmp/s7_smoke; exit 1; }
rm -rf /tmp/s7_smoke /tmp/s7_smoke_ad /root/hf/hub/models--Qwen--Qwen3-0.6B
echo "[smoke] ok"

if [ "$MODE" = pilot ]; then
  $PY $S7/s7_run.py --seed 1 --stop-after 0.1 --out $OUT --adapter-dir $ADIR 2>&1 | tee -a $OUT/run.log
else
  for SEED in 1 2; do
    $PY $S7/s7_run.py --seed $SEED --out $OUT --adapter-dir $ADIR 2>&1 | tee -a $OUT/run.log
  done
fi

# wait for the local puller (heartbeat every 30 s keeps the launcher's stall watchdog quiet)
N=$(find $ADIR -maxdepth 1 -name "*.done" | wc -l)
for i in $(seq 1 $(( ${WAIT_MIN:-25} * 2 ))); do
  P=$(find $ADIR -maxdepth 1 -name "*.pulled" | wc -l)
  [ "$P" -ge "$N" ] && break
  echo "[adapters] $P/$N pulled by the local puller; waiting ($i)"
  sleep 30
done
for d in $ADIR/*.done; do
  [ -e "$d" ] || continue
  name=$(basename "$d" .done)
  if [ ! -f "$ADIR/$name.pulled" ]; then
    mkdir -p $OUT/adapters_unpulled && cp -r "$ADIR/$name" "$d" $OUT/adapters_unpulled/
    echo "[adapters] $name not pulled; left in results for the launcher's pull"
  fi
done
du -sh $ADIR $OUT | tee -a $OUT/run.log
echo "[s7] $MODE done"
