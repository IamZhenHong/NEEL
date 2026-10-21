#!/usr/bin/env bash
# Tier 2 turn anchors (DESIGN.md) into persona/reasoning/${DIR:?}
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=persona/reasoning/${DIR:?}
mkdir -p $OUT
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1 || { tail -30 $OUT/setup_vllm.log; echo "[tier2] vLLM setup FAILED"; exit 1; }
/root/vllm/bin/python -m persona.reasoning.tier2 anchors --out $OUT 2>&1 | tee -a $OUT/run.log; rc=${PIPESTATUS[0]}
echo "[tier2] pod done (exit $rc)" | tee -a $OUT/run.log
exit $rc
