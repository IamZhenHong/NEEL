#!/usr/bin/env bash
# s2 manual chain (after the pilot-driven fixes; pod.sh is paused): Gemma pilot2; Qwen pilot2, then the frozen main
# file from /root/s2_inbox/qwen2; Gemma main from /root/s2_inbox/gemma2. STOP files end each waiting process.
cd /root/NEEL
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/root/hf PYTHONUNBUFFERED=1
export HF_TOKEN=$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^HF_TOKEN=//p')
E=selfknow/s2_role_wins; R=$E/results
PY="/root/vllm/bin/python -m selfknow.s2_role_wins.pod_run --out $R --max-model-len 8192"
mkdir -p /root/s2_inbox/qwen2 /root/s2_inbox/gemma2
echo "[s2-manual] start $(date -u +%H:%M:%S); identities sha $(sha256sum $E/identities.py | cut -c1-12)"
$PY --model google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --fmt gemma4 --tag gemma --initial $E/jobs/pilot2_gemma.jsonl
$PY --model Qwen/Qwen3.6-27B --revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 --fmt qwen --tag qwen --initial $E/jobs/pilot2_qwen.jsonl --inbox /root/s2_inbox/qwen2 --wait-minutes 20
$PY --model google/gemma-4-31B-it --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --fmt gemma4 --tag gemma --inbox /root/s2_inbox/gemma2 --wait-minutes 15
echo "[s2-manual] done $(date -u +%H:%M:%S)"
