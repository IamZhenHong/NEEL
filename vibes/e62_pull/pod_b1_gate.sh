#!/usr/bin/env bash
# e62b Amendment B1, gate-1 retry only: M2's 80 pilot answers (8 x 10) with an explicit empty system prompt (vLLM).
#   uv run python infra/launch_run.py --name e62b-2 ... --results-dir vibes/e62_pull/results/b1 -- bash vibes/e62_pull/pod_b1_gate.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export E62_VARIANT=b E62B_B1=1
E=vibes/e62_pull
P=$E/results/b1/pilot; mkdir -p $P
VPY="env VLLM_USE_FLASHINFER_SAMPLER=0 /root/vllm/bin/python"
log() { echo "[e62b-B1 $(date -u +%H:%M:%S)] $*" | tee -a $P/run.log; }
log "B1 gate-1 retry, rev ${GIT_REV:-?}"
bash $E/setup_vllm.sh 2>&1 | tee -a $P/run.log
$VPY -m vibes.e62_pull.gen_vllm --model REF --mode main --out $P --pilot-out $P 2>&1 | tee -a $P/run.log
head -c 600 $P/replies_em/M.jsonl | tee -a $P/run.log; echo
log "done"
