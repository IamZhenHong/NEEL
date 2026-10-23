#!/usr/bin/env bash
# PA-R2C pod side: build the vLLM venv, then run q36 and g31 in turn (one vLLM process each) into $OUT/<model>/.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-persona_affect/r2c_care/results/main}
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh > "$OUT/setup_vllm.log" 2>&1 || { tail -30 "$OUT/setup_vllm.log"; echo "[r2c] vLLM setup FAILED"; exit 1; }
rc=0
for m in q36 g31; do
  /root/vllm/bin/python -m persona_affect.r2c_care.r2c run --model "$m" --out "$OUT/$m" 2>&1 | tee -a "$OUT/run.log"
  r=${PIPESTATUS[0]}; [ "$r" -ne 0 ] && rc=$r
done
echo "[r2c] pod done (exit $rc)" | tee -a "$OUT/run.log"
exit $rc
