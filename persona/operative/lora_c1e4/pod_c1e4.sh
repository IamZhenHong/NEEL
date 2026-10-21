#!/usr/bin/env bash
# C1 + E4 pod (one A100 80GB; lora_c1e4/DESIGN.md). vLLM setup runs in the background while P and C1 train (project
# venv); then the base phase (vLLM: own-words harvest, E4 rows, base eval); E4 trains; per arm: merge + check, eval
# (vLLM), delete the merged copy. Arms P, C1, E4, E4d (E4 with disowned records). Local dry runs: PY / VPY = local
# pythons, SKIP_SETUP=1, TRAIN_MODEL / MERGE_BASE = a tiny checkpoint, a fake vllm on PYTHONPATH.
set -uo pipefail
cd "$(dirname "$0")/../../.."
L=persona/operative/lora_c1e4; OUT=$L/results/pod; AD=$OUT/adapters
mkdir -p $OUT $AD
export VLLM_USE_FLASHINFER_SAMPLER=0
PY=${PY:-/root/venv/bin/python}; VPY=${VPY:-/root/vllm/bin/python}
TM=${TRAIN_MODEL:-Qwen/Qwen3.5-9B}; MB=${MERGE_BASE:+--base $MERGE_BASE}
for f in C1.jsonl P.jsonl ih_attacks.jsonl ih_tasks.jsonl.gz dec_pool_9b.json; do
  test -s $L/data/$f || { echo "[c1e4] data/$f missing"; exit 1; }
done
if [ -z "${SKIP_SETUP:-}" ]; then
  ( bash vigilante/pod/setup_vllm.sh > $OUT/setup_vllm.log 2>&1; echo $? > /root/vllm_setup.rc ) &
  uv pip install -q --python $PY flash-linear-attention > $OUT/fla.log 2>&1 && echo "[c1e4] fla installed" \
    || echo "[c1e4] fla install failed; reference kernels"
fi
$PY -m persona.operative.lora_c1e4.train train --arms P C1 --model $TM --e4 none --out $AD 2>&1 | tee -a $OUT/train.log
rc=${PIPESTATUS[0]}; [ $rc = 0 ] || { echo "[c1e4] training P/C1 failed ($rc)"; exit $rc; }
if [ -z "${SKIP_SETUP:-}" ]; then
  for i in $(seq 1 90); do [ -f /root/vllm_setup.rc ] && break; echo "[c1e4] waiting for vLLM setup"; sleep 20; done
  [ "$(cat /root/vllm_setup.rc 2>/dev/null)" = 0 ] || { tail -30 $OUT/setup_vllm.log; echo "[c1e4] vLLM setup FAILED"; exit 1; }
fi
$VPY -m persona.operative.lora_c1e4.pod base --out $OUT 2>&1 | tee -a $OUT/run.log
rc=${PIPESTATUS[0]}; [ $rc = 0 ] || { echo "[c1e4] base phase failed ($rc)"; exit $rc; }
$PY -m persona.operative.lora_c1e4.train train --arms E4 E4d --model $TM --e4 $OUT --out $AD 2>&1 | tee -a $OUT/train.log
rc=${PIPESTATUS[0]}; [ $rc = 0 ] || { echo "[c1e4] training E4 / E4d failed ($rc)"; exit $rc; }
for arm in P C1 E4 E4d; do
  M=${MERGED:-/root/merged}/$arm
  $PY -m persona.operative.lora_c1e4.train merge --arm $arm --model $TM $MB --adapters $AD --out $M 2>&1 | tee -a $OUT/merge.log
  rc=${PIPESTATUS[0]}; [ $rc = 0 ] || { echo "[c1e4] merge $arm failed ($rc)"; exit $rc; }
  cp $M/merge_check.json $OUT/merge_check_$arm.json
  $VPY -m persona.operative.lora_c1e4.pod eval --arm $arm --path $M --out $OUT 2>&1 | tee -a $OUT/run.log
  rc=${PIPESTATUS[0]}; [ $rc = 0 ] || { echo "[c1e4] eval $arm failed ($rc)"; exit $rc; }
  rm -rf $M
done
echo "[c1e4] pod done" | tee -a $OUT/run.log
