#!/usr/bin/env bash
# PA-P9 pod (one A100 80GB): vLLM setup, then pod_self.py into OUT (persona_affect/p9_bakeoff/results/self), which the
# launcher pulls back. Launch with launch.sh, never by hand.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-persona_affect/p9_bakeoff/results/self}
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0
bash vigilante/pod/setup_vllm.sh 2>&1 | tail -5 | tee "$OUT/setup.log"
[ "${PIPESTATUS[0]}" = 0 ] || { echo "[p9] vLLM setup failed"; exit 1; }
"${VPY:-/root/vllm/bin/python}" -m persona_affect.p9_bakeoff.pod_self --out "$OUT" 2>&1 | tee "$OUT/run.log"
exit "${PIPESTATUS[0]}"
