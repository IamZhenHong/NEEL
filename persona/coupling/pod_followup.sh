#!/usr/bin/env bash
# Stage 1 follow-up pod (exploratory; PLAN.md "Stage 1 follow-up"): train B and C at seed 2, run the seed-2 gate cells
# (no harm cells), then the manipulation checks on all seed-1 and seed-2 adapters. Seed-1 adapters arrive via sync.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
C=persona/coupling; OUT=$C/results/stage1_s2; AD=$C/results/adapters
mkdir -p $OUT $AD $C/results/manip
export VLLM_USE_FLASHINFER_SAMPLER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for arm in A B Bp C; do test -s $AD/${arm}_s1/adapter_model.safetensors || { echo "[cplf] ${arm}_s1 missing"; exit 1; }; done
( bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1; echo $? > /root/vllm_setup.rc ) &
$PY -m persona.coupling.train --arms B C --seed 2 --micro-batch 4 --out $AD 2>&1 | tee -a $OUT/train.log
for arm in B C; do test -s $AD/${arm}_s2/adapter_model.safetensors || { echo "[cplf] training $arm failed"; exit 1; }; done
for i in $(seq 1 60); do [ -f /root/vllm_setup.rc ] && break; echo "[cplf] waiting for vLLM setup"; sleep 20; done
[ "$(cat /root/vllm_setup.rc 2>/dev/null)" = 0 ] || { tail -30 $OUT/setup_vllm.log; echo "[cplf] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.coupling.stage0 run --jobs $C/jobs/stage1_s2_gate.jsonl --adapter-dir $AD --out $OUT 2>&1 | tee -a $OUT/run.log
/root/vllm/bin/python -m persona.coupling.manip run --adapter-dir $AD --out $C/results/manip 2>&1 | tee -a $C/results/manip/run.log
echo "[cplf] pod done" | tee -a $OUT/run.log
