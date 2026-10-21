#!/usr/bin/env bash
# Seeds 17 + 20 (DESIGN.md) into persona/reasoning/${DIR:?}/{s17,s20}
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/reasoning/${DIR:?}
mkdir -p $OUT/s17 $OUT/s20
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[s1720] vLLM setup FAILED"; exit 1; }
python -m persona.reasoning.seed1720 merge --adapters persona/operative/lora_c1e4/results/pod/adapters --merged /root/merged 2>&1 | tee -a $OUT/run.log
rc=${PIPESTATUS[0]}; [ $rc -ne 0 ] && { echo "[s1720] merge FAILED"; exit $rc; }
for ARM in P C1 E4; do
  /root/vllm/bin/python -m persona.reasoning.seed1720 gen --arm $ARM --merged /root/merged --out $OUT/s17 2>&1 | tee -a $OUT/run.log
done
python -m persona.reasoning.seed1720 s20 --merged /root/merged --out $OUT/s20 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[s1720] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
