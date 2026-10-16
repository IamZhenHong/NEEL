#!/usr/bin/env bash
# e62b pod job (DESIGN_b.md): released full-ft EM organism M2 = ModelOrganismsForEM/Qwen2.5-14B-Instruct_full-ft,
# base B2 = Qwen/Qwen2.5-14B-Instruct, one A100 80GB. Two off-pod decision pauses:
#   gate 1 (pilot): M2's 80 pilot answers (8 x 10, separate seeds) -> results/b/pilot; judged off-pod (Haiku + gpt-5-mini);
#                   /root/e62b_go1 continues (after e62b-freeze), /root/e62b_stop1 or 25 min silence stops.
#   gate 2 (sturdiness): after f (value-free control SFT) at full n, R_f100 is judged off-pod (Haiku);
#                   /root/e62b_go2 trains b and e; /root/e62b_stop2 or 25 min silence skips them (card: stop).
# B2's and M2's main answers (and a2's) are written before gate 1 to save two vLLM start-ups; not read before the freeze.
# KL/MMLU (HF) run at the end for every dose that exists.
#   uv run python infra/launch_run.py --name e62b-<n> ... --results-dir vibes/e62_pull/results/b -- bash vibes/e62_pull/pod_b.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export E62_VARIANT=b
E=vibes/e62_pull
OUT=$E/results/b/main; P=$E/results/b/pilot; WORK=/root/e62b; mkdir -p $OUT $P $WORK
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
VPY="env VLLM_USE_FLASHINFER_SAMPLER=0 E62_VARIANT=b /root/vllm/bin/python"
log() { echo "[e62b $(date -u +%H:%M:%S)] $*" | tee -a $OUT/run.log; }
wait_for() {   # $1 = gate number; returns 0 on go, 1 on stop or timeout
  local n=$1 i
  log "gate $n: waiting for /root/e62b_go$n or /root/e62b_stop$n"
  for i in $(seq 1 150); do
    if [ -f /root/e62b_go$n ]; then log "gate $n: go"; return 0; fi
    if [ -f /root/e62b_stop$n ] || [ "$i" = 150 ]; then log "gate $n: stop"; return 1; fi
    if [ $((i % 6)) = 0 ]; then log "gate $n: waiting ($((i / 6)) min)"; fi
    sleep 10
  done
}

log "e62b start, rev ${GIT_REV:-?}"
bash $E/setup_vllm.sh 2>&1 | tee -a $OUT/run.log

# ---- B2: SFT data (b, e, f; e62 prompts and settings) + B2's EM answers (vLLM)
$VPY -m vibes.e62_pull.gen_vllm --sft --mode main --out $OUT --doses B 2>&1 | tee -a $OUT/run.log
$PY -m vibes.e62_pull.match_tokens --raw $OUT/sft_raw --out $OUT/sft 2>&1 | tee -a $OUT/run.log
log "B2 done"

# ---- M2: EM answers (main), a2 (constitution), and the gate-1 pilot answers (vLLM)
$VPY -m vibes.e62_pull.gen_vllm --model REF --mode main --out $OUT --doses M a2 --pilot-out $P 2>&1 | tee -a $OUT/run.log
log "M2 done; gate-1 answers in $P/replies_em/M.jsonl"
wait_for 1 || { log "stopped at gate 1"; exit 0; }

# ---- gate 2: the value-free control f first
$PY -m vibes.e62_pull.train --out $OUT --work $WORK --sft-dir $OUT/sft --skip-m --techs f 2>&1 | tee -a $OUT/run.log
$VPY -m vibes.e62_pull.gen_vllm --model REF --lora-dir $WORK/lora --mode main --out $OUT --doses f25 f50 f100 2>&1 \
  | tee -a $OUT/run.log
log "f done; gate-2 answers in $OUT/replies_em/f100.jsonl"
if wait_for 2; then
  $PY -m vibes.e62_pull.train --out $OUT --work $WORK --sft-dir $OUT/sft --skip-m --techs b e 2>&1 | tee -a $OUT/run.log
  $VPY -m vibes.e62_pull.gen_vllm --model REF --lora-dir $WORK/lora --mode main --out $OUT \
    --doses b25 b50 b100 e25 e50 e100 2>&1 | tee -a $OUT/run.log
  log "b, e done"
else
  log "gate 2 stop: b and e not trained"
fi

# ---- KL from M2, MMLU for every dose that exists; B2 last (HF)
$PY -m vibes.e62_pull.hf_eval --out $OUT --work $WORK --mode main 2>&1 | tee -a $OUT/run.log
log "pod done"
