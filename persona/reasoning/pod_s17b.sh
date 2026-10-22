#!/usr/bin/env bash
# Seed 17b (DESIGN.md) into persona/reasoning/${DIR:?}: Pself targets (vLLM, base 9B) -> same-pod base anchor eval ->
# LoRA training + checked merge (lora_c1e4/train.py, unchanged) -> Pself eval.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/reasoning/${DIR:?}
mkdir -p $OUT/s17 $OUT/adapters
export VLLM_USE_FLASHINFER_SAMPLER=0
PY=/root/venv/bin/python; VPY=/root/vllm/bin/python
( bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1; echo $? > /root/vllm_setup.rc ) &
uv pip install -q --python $PY flash-linear-attention > $OUT/fla.log 2>&1 && echo "[s17b] fla installed" \
  || echo "[s17b] fla install failed; reference kernels"
for i in $(seq 1 90); do [ -f /root/vllm_setup.rc ] && break; echo "[s17b] waiting for vLLM setup"; sleep 20; done
[ "$(cat /root/vllm_setup.rc 2>/dev/null)" = 0 ] || { tail -30 $OUT/setup_vllm.log; echo "[s17b] vLLM setup FAILED"; exit 1; }
$VPY -m persona.reasoning.s17b targets --out $OUT 2>&1 | tee -a $OUT/run.log
rc=${PIPESTATUS[0]}; [ $rc = 0 ] || { echo "[s17b] targets FAILED ($rc)"; exit $rc; }
$VPY -m persona.reasoning.seed1720 gen --arm base --out $OUT/s17 2>&1 | tee -a $OUT/run.log
rc=${PIPESTATUS[0]}; [ $rc = 0 ] || { echo "[s17b] base gen FAILED ($rc)"; exit $rc; }
$PY -m persona.operative.lora_c1e4.train train --arms Pself --data $OUT --e4 none --out $OUT/adapters 2>&1 | tee -a $OUT/train.log
rc=${PIPESTATUS[0]}; [ $rc = 0 ] || { echo "[s17b] training FAILED ($rc)"; exit $rc; }
$PY -m persona.operative.lora_c1e4.train merge --arm Pself --adapters $OUT/adapters --out /root/merged/Pself 2>&1 | tee -a $OUT/merge.log
rc=${PIPESTATUS[0]}; [ $rc = 0 ] || { echo "[s17b] merge FAILED ($rc)"; exit $rc; }
cp /root/merged/Pself/merge_check.json $OUT/merge_check_Pself.json
$VPY -m persona.reasoning.seed1720 gen --arm Pself --merged /root/merged --out $OUT/s17 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[s17b] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
