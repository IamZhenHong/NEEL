#!/usr/bin/env bash
# Stage 1 pod (one A100 80GB): train A, B, Bp, C (seed $SEED) in the project venv, then vLLM: base + the four adapters
# on the stage-1 jobs. vLLM setup runs in the background during training.
set -uo pipefail
cd "$(dirname "$0")/../.."
SEED=${SEED:-1}
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
C=persona/coupling; OUT=$C/results/stage1_s$SEED; AD=$C/results/adapters
mkdir -p $OUT $AD
export VLLM_USE_FLASHINFER_SAMPLER=0
for arm in A B Bp C; do test -s $C/data/$arm.jsonl || { echo "[cpl1] data $arm missing"; exit 1; }; done
( bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1; echo $? > /root/vllm_setup.rc ) &
# smoke: tiny training on Qwen3-0.6B
$PY -m persona.coupling.train --arms A --model Qwen/Qwen3-0.6B --n-cap 16 --out /tmp/cpl_tiny_ad > /tmp/cpl_tiny_ad.log 2>&1 \
  && test -s /tmp/cpl_tiny_ad/A_s1/adapter_model.safetensors || { tail -40 /tmp/cpl_tiny_ad.log; echo "[smoke] train FAILED"; exit 1; }
echo "[smoke] train ok" | tee -a $OUT/run.log
TODO=$(for arm in A B Bp C; do test -s $AD/${arm}_s$SEED/adapter_model.safetensors || echo $arm; done | xargs)
echo "[cpl1] arms to train: ${TODO:-none (synced)}" | tee -a $OUT/run.log
[ -z "$TODO" ] || $PY -m persona.coupling.train --arms $TODO --seed $SEED --out $AD 2>&1 | tee -a $OUT/train.log
for arm in A B Bp C; do test -s $AD/${arm}_s$SEED/adapter_model.safetensors || { echo "[cpl1] training $arm failed"; exit 1; }; done
for i in $(seq 1 60); do [ -f /root/vllm_setup.rc ] && break; echo "[cpl1] waiting for vLLM setup"; sleep 20; done
[ "$(cat /root/vllm_setup.rc 2>/dev/null)" = 0 ] || { tail -30 $OUT/setup_vllm.log; echo "[cpl1] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.coupling.stage0 run --jobs $C/jobs/stage1_s$SEED.jsonl --adapter-dir $AD --out $OUT 2>&1 | tee -a $OUT/run.log
echo "[cpl1] pod done" | tee -a $OUT/run.log
