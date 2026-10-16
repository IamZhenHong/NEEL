#!/usr/bin/env bash
# e62 pod job (Qwen3-14B, one A100 80GB). Usage: bash vibes/e62_pull/pod.sh pilot|main
#   pilot: B writes the SFT data (built into token-matched sets on the pod), B EM answers; train M + b/e/f; all doses
#          at pilot n (M 10 samples/question, other doses 2; 10 KL prompts; 20 MMLU items)
#   main:  the committed SFT sets (vibes/e62_pull/materials/sft_*.jsonl); all doses at full n
# Launch (from repo root, see README):
#   uv run python infra/launch_run.py --name e62-<n> --allow-concurrent --max-minutes M --max-cost C \
#     --gpus "NVIDIA A100-SXM4-80GB" "NVIDIA A100 80GB PCIe" --container-gb 160 --exclude ... \
#     --results-dir vibes/e62_pull/results -- bash vibes/e62_pull/pod.sh pilot
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-pilot}
E=vibes/e62_pull
OUT=$E/results/$MODE; WORK=/root/e62; mkdir -p $OUT $WORK
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
VPY="env VLLM_USE_FLASHINFER_SAMPLER=0 /root/vllm/bin/python"
log() { echo "[e62 $(date -u +%H:%M:%S)] $*" | tee -a $OUT/run.log; }
DOSES_V="M a1 a2 b12p5 b25 b50 b100 e12p5 e25 e50 e100 f12p5 f25 f50 f100"

log "mode $MODE, rev ${GIT_REV:-?}"
bash $E/setup_vllm.sh 2>&1 | tee -a $OUT/run.log

# ---- smoke: the whole chain on Qwen3-0.6B; any failure stops before the paid part
S=/root/e62_smoke; rm -rf $S; mkdir -p $S
log "smoke start"
if $VPY -m vibes.e62_pull.gen_vllm --tiny --sft --out $S/res --doses B > $S/log 2>&1 \
   && $PY -m vibes.e62_pull.match_tokens --tiny --raw $S/res/sft_raw --out $S/mat >> $S/log 2>&1 \
   && $PY -m vibes.e62_pull.train --tiny --out $S/res --work $S/work --sft-dir $S/mat >> $S/log 2>&1 \
   && $VPY -m vibes.e62_pull.gen_vllm --tiny --model $S/work/M --lora-dir $S/work/lora --mode $MODE --out $S/res \
        --doses M a2 b12p5 e100 f50 >> $S/log 2>&1 \
   && $PY -m vibes.e62_pull.hf_eval --tiny --out $S/res --work $S/work --mode $MODE >> $S/log 2>&1 \
   && test -s $S/res/kl.json && test -s $S/res/replies_em/c8.jsonl && test -s $S/res/replies_em/e100.jsonl \
   && test -s $S/res/replies_em/B.jsonl && grep -q '"B"' $S/res/kl.json; then
  log "smoke ok"
else
  tail -60 $S/log; log "smoke FAILED"; cp $S/log $OUT/smoke_failed.log; exit 1
fi
rm -rf $S

# ---- B: SFT data (pilot only) and B's EM answers (vLLM)
if [ "$MODE" = pilot ]; then
  $VPY -m vibes.e62_pull.gen_vllm --sft --mode pilot --out $OUT --doses B 2>&1 | tee -a $OUT/run.log
  $PY -m vibes.e62_pull.match_tokens --raw $OUT/sft_raw --out $OUT/sft 2>&1 | tee -a $OUT/run.log
  SFT=$OUT/sft
else
  $VPY -m vibes.e62_pull.gen_vllm --mode main --out $OUT --doses B 2>&1 | tee -a $OUT/run.log
  SFT=$E/materials
fi
log "B done"

# ---- M (axis stats on B, EM LoRA, merge) and b/e/f LoRAs (HF)
$PY -m vibes.e62_pull.train --out $OUT --work $WORK --sft-dir $SFT 2>&1 | tee -a $OUT/run.log
log "training done"

# ---- prompt and LoRA doses + M (vLLM on merged M)
$VPY -m vibes.e62_pull.gen_vllm --model $WORK/M --lora-dir $WORK/lora --mode $MODE --out $OUT --doses $DOSES_V 2>&1 \
  | tee -a $OUT/run.log
log "vLLM doses done"

# ---- KL, MMLU at every dose; hook doses' EM answers; B anchor (HF)
$PY -m vibes.e62_pull.hf_eval --out $OUT --work $WORK --mode $MODE 2>&1 | tee -a $OUT/run.log
du -sh $WORK/lora 2>/dev/null | tee -a $OUT/run.log
log "pod done"
