#!/usr/bin/env bash
# e61 main pod job (one A100 80GB), via launch.sh <name> main ... (after e61-freeze and vb-freeze).
#   1. smoke on Qwen3-0.6B: tiny LoRA training + HF reads, then the vLLM evaluator with real LoRA requests;
#   2. train the five LoRAs at the pilot's setting (LR, EPOCHS from env) and read every condition (HF, pod_hf.py main);
#   3. one vLLM engine: VB-1 batteries, free-form + benign replies, Yes/No engine check, for all six conditions.
# Adapters stay on the pod (adapters_pod/main); results/main is pulled by the launcher.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=vibes/e61_free_vars
R=$E/results/main
LR=${LR:?set LR}; EPOCHS=${EPOCHS:?set EPOCHS}
mkdir -p $R
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
log() { echo "$@" | tee -a $R/run.log; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $R/run.log
$PY -c "import torch, transformers, peft; print('[versions] torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee -a $R/run.log
bash $E/setup_vllm.sh 2>&1 | tee -a $R/run.log
[ ${PIPESTATUS[0]} = 0 ] || { log "[e61] vLLM setup failed"; exit 1; }
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
VPY=/root/vllm/bin/python
sha256sum $E/data/train/*.jsonl | tee -a $R/run.log

# ---------------------------------------------------------------- smoke (tiny model, real vLLM LoRA path)
S=/tmp/e61_smoke; rm -rf $S; mkdir -p $S
( $PY $E/pod_hf.py main --lr 1e-4 --epochs 1 --model Qwen/Qwen3-0.6B --tiny --out $S/main --adapters $S/ad \
  && $VPY $E/eval_vllm.py --model Qwen/Qwen3-0.6B --tiny --adapters $S/ad --out $S/eval ) > $S/smoke.log 2>&1 \
  || { tail -40 $S/smoke.log; log "[smoke] FAILED"; exit 1; }
test -s $S/main/reads.jsonl && test -s $S/eval/vb_outputs.jsonl && test -s $S/eval/agree_vllm.jsonl || { ls -R $S; log "[smoke] missing outputs"; exit 1; }
grep "\[eval\]" $S/smoke.log | tee -a $R/run.log
rm -rf $S /root/hf/hub/models--Qwen--Qwen3-0.6B
log "[smoke] ok"

# ---------------------------------------------------------------- main
$PY $E/pod_hf.py main --lr $LR --epochs $EPOCHS --out $R --adapters $E/adapters_pod/main 2>&1 | tee -a $R/run.log
[ ${PIPESTATUS[0]} = 0 ] || { log "[e61] training/reads failed"; exit 1; }
$VPY $E/eval_vllm.py --adapters $E/adapters_pod/main --out $R 2>&1 | tee -a $R/run.log
[ ${PIPESTATUS[0]} = 0 ] || { log "[e61] vLLM evaluation failed"; exit 1; }
du -sh $E/adapters_pod/main $R | tee -a $R/run.log
log "[e61] main done"
