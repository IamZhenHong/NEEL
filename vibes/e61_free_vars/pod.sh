#!/usr/bin/env bash
# e61 pod job (one A100 80GB), via launch.sh -> infra/launch_run.py.
#   MODE=pilot: vLLM self-generation (training replies + untouched base-rate replies) -> wait for the local relay
#               (relay.py: Haiku pairing filter + register match, uploads data/train/*.jsonl and READY) -> T1 LR pilot.
#   MODE=main:  defined in pod_main.sh (written after vb-freeze).
# A Qwen3-0.6B smoke (vLLM generation, stub pairing, HF pilot) runs first so a bug stops the job before the paid part.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=vibes/e61_free_vars
MODE=${MODE:?set MODE=pilot}
R=$E/results/$MODE
mkdir -p $R $E/data
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
log() { echo "$@" | tee -a $R/run.log; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/run.log
$PY -c "import torch, transformers, peft; print('[versions] torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee -a $R/run.log
bash $E/setup_vllm.sh 2>&1 | tee -a $R/run.log
[ ${PIPESTATUS[0]} = 0 ] || { log "[e61] vLLM setup failed"; exit 1; }
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
VPY=/root/vllm/bin/python

# ---------------------------------------------------------------- smoke (tiny model)
S=/tmp/e61_smoke; rm -rf $S; mkdir -p $S
( $VPY $E/gen.py data --engine vllm --model Qwen/Qwen3-0.6B --tiny --out $S/gen_raw.jsonl --untouched-eval-out $S/ue.jsonl \
  && $PY $E/pairing.py --raw $S/gen_raw.jsonl --out $S --stub --tiny --no-validate \
  && $PY $E/pod_hf.py pilot --model Qwen/Qwen3-0.6B --tiny --data-dir $S/train --out $S/pilot --adapters $S/ad ) > $S/smoke.log 2>&1 \
  || { tail -40 $S/smoke.log; log "[smoke] FAILED"; exit 1; }
test -s $S/ue.jsonl && test -s $S/pilot/pilot.json && test -s $S/train/C.jsonl || { ls -R $S; log "[smoke] missing outputs"; exit 1; }
grep "\[gen\] wrote" $S/smoke.log | tee -a $R/run.log
rm -rf $S /root/hf/hub/models--Qwen--Qwen3-0.6B
log "[smoke] ok"

if [ "$MODE" = pilot ]; then
  # 1. self-generated training replies + untouched base-rate replies (one vLLM engine)
  $VPY $E/gen.py data --engine vllm --out $E/data/gen_raw.jsonl --untouched-eval-out $R/untouched_eval.jsonl 2>&1 | tee -a $R/run.log
  [ ${PIPESTATUS[0]} = 0 ] && test -s $E/data/gen_raw.jsonl || { log "[e61] generation failed"; exit 1; }
  cp $E/data/gen_raw.jsonl $R/gen_raw.jsonl
  touch $E/data/gen.done
  log "[e61] generation done; waiting for the relay's training sets"
  # 2. wait for the local relay (heartbeat every 30 s keeps the launcher's stall watchdog quiet)
  for i in $(seq 1 $(( ${WAIT_MIN:-40} * 2 ))); do
    [ -f $E/data/train/READY ] && break
    echo "[wait] relay: pairing filter not uploaded yet ($i)"
    sleep 30
  done
  [ -f $E/data/train/READY ] || { log "[e61] no training sets from the relay after ${WAIT_MIN:-40} min"; exit 1; }
  sha256sum $E/data/train/*.jsonl | tee -a $R/run.log
  # 3. T1 LR pilot (card: first of (3e-5,1), (1e-4,1), (1e-4,2) that passes the value gate)
  $PY $E/pod_hf.py pilot --out $R --adapters $E/adapters_pod/pilot 2>&1 | tee -a $R/run.log
  [ ${PIPESTATUS[0]} = 0 ] || { log "[e61] pilot failed"; exit 1; }
  log "[e61] pilot done"
fi
