#!/usr/bin/env bash
# Seeds 10 (JOB=q11e, vLLM) or 4/7/8 (JOB=probes, transformers in the project venv) into persona/reasoning/${DIR:?}
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/reasoning/${DIR:?}
mkdir -p $OUT
if [ "${JOB:?}" = q11e ]; then
  export VLLM_USE_FLASHINFER_SAMPLER=0
  bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[seeds] vLLM setup FAILED"; exit 1; }
  /root/vllm/bin/python -m persona.reasoning.q11e run --out $OUT 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
else
  python -m persona.reasoning.probes run --out $OUT --seeds ${SEEDS:-4,7,8} 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
fi
echo "[seeds] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
