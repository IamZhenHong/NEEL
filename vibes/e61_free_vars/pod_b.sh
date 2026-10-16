#!/usr/bin/env bash
# e61b pod job (one A100 80GB), via launch.sh <name> b ...:
#   1. smoke on Qwen3-0.6B (tiny LoRAs + HF reads, eval_b with real vLLM LoRA path, both system prompts);
#   2. gate pilot: untouched Qwen3-14B under theatrical, 12 seeded HC items x 2 samples -> results/b/pilot, PILOT.done;
#   3. wait for the local side (relay_b.py: judge the pilot, freeze e61b-freeze) to upload GO (or STOP);
#   4. main: retrain the five LoRAs (e61 recipe: 1e-4, 2 epochs, seed 1) + HF Yes/No reads; then VB-1 for all six
#      conditions under theatrical and under no system prompt, plus the vLLM Yes/No engine check.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=vibes/e61_free_vars
R=$E/results/b
mkdir -p $R/pilot
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
log() { echo "$@" | tee -a $R/run.log; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/run.log
$PY -c "import torch, transformers, peft; print('[versions] torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee -a $R/run.log
bash $E/setup_vllm.sh 2>&1 | tee -a $R/run.log
[ ${PIPESTATUS[0]} = 0 ] || { log "[e61b] vLLM setup failed"; exit 1; }
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
VPY=/root/vllm/bin/python
sha256sum $E/data/train/*.jsonl | tee -a $R/run.log

# ---------------------------------------------------------------- smoke
S=/tmp/e61b_smoke; rm -rf $S; mkdir -p $S
( $PY $E/pod_hf.py main --lr 1e-4 --epochs 1 --model Qwen/Qwen3-0.6B --tiny --out $S/main --adapters $S/ad \
  && $VPY $E/eval_b.py --model Qwen/Qwen3-0.6B --tiny --adapters $S/ad --out $S/eval ) > $S/smoke.log 2>&1 \
  || { tail -40 $S/smoke.log; log "[smoke] FAILED"; exit 1; }
test -s $S/main/reads.jsonl && test -s $S/eval/vb_outputs.jsonl && test -s $S/eval/agree_vllm.jsonl || { ls -R $S; log "[smoke] missing outputs"; exit 1; }
grep "\[eval_b\]" $S/smoke.log | tee -a $R/run.log
rm -rf $S /root/hf/hub/models--Qwen--Qwen3-0.6B
log "[smoke] ok"

# ---------------------------------------------------------------- gate pilot
$VPY $E/eval_b.py --pilot-hc 12 --out $R/pilot 2>&1 | tee -a $R/run.log
[ ${PIPESTATUS[0]} = 0 ] || { log "[e61b] pilot failed"; exit 1; }
touch $R/pilot/PILOT.done
log "[e61b] pilot done; waiting for GO"
for i in $(seq 1 $(( ${WAIT_MIN:-30} * 2 ))); do
  [ -f $R/GO ] || [ -f $R/STOP ] && break
  echo "[wait] freeze: no GO yet ($i)"
  sleep 30
done
[ -f $R/STOP ] && { log "[e61b] STOP received"; exit 0; }
[ -f $R/GO ] || { log "[e61b] no GO after ${WAIT_MIN:-30} min"; exit 1; }
log "[e61b] GO ($(cat $R/GO))"

# ---------------------------------------------------------------- main
$PY $E/pod_hf.py main --lr 1e-4 --epochs 2 --out $R --adapters $E/adapters_pod/b 2>&1 | tee -a $R/run.log
[ ${PIPESTATUS[0]} = 0 ] || { log "[e61b] training/reads failed"; exit 1; }
$VPY $E/eval_b.py --adapters $E/adapters_pod/b --out $R 2>&1 | tee -a $R/run.log
[ ${PIPESTATUS[0]} = 0 ] || { log "[e61b] evaluation failed"; exit 1; }
du -sh $E/adapters_pod/b $R | tee -a $R/run.log
log "[e61b] main done"
