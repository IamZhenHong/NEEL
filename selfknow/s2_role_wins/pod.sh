#!/usr/bin/env bash
# s2 on one A100 80GB, two models as separate processes, thinking off.
#   1. Gemma 4 31B: pilot (jobs/pilot_gemma.jsonl), exit.
#   2. Qwen3.6-27B: pilot (jobs/pilot_qwen.jsonl), then stay loaded and wait (<= 25 min) for job files in
#      /root/s2_inbox/qwen. The main job file is uploaded there only after the pilot is recorded in README.md and the
#      s2-freeze tag is set locally; STOP ends the process.
#   3. Gemma 4 31B again: waits for /root/s2_inbox/gemma (main job file or STOP) before loading, runs it, exits.
# Both models are pre-downloaded (pinned revisions, Gemma first) while vLLM installs.
set -uo pipefail
cd "$(dirname "$0")/../.."
E=selfknow/s2_role_wins
R=$E/results
IN=/root/s2_inbox
mkdir -p $R $IN/qwen $IN/gemma
QREV=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
GREV=842da3794eaa0b77d5f08bae87a17459d91ff475
(
  /root/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('google/gemma-4-31B-it', revision='$GREV', max_workers=16)" && touch /root/dl_gemma.ok
  /root/venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.6-27B', revision='$QREV', max_workers=16)" && touch /root/dl_qwen.ok
) > $R/download.log 2>&1 &
bash vigilante/pod/setup_vllm.sh 2>&1 | tee -a $R/pod.log
export PATH=/root/vllm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
waitfor() { for i in $(seq 1 120); do [ -f "$1" ] && { echo "[s2] $1 present"; return 0; }; sleep 10; [ $((i % 6)) = 0 ] && echo "[s2] waiting for $1 ($((i * 10))s)"; done; echo "[s2] $1 never appeared; vLLM will download"; return 0; }
PY="/root/vllm/bin/python -m selfknow.s2_role_wins.pod_run --out $R --max-model-len 8192"

waitfor /root/dl_gemma.ok
$PY --model google/gemma-4-31B-it --revision $GREV --fmt gemma4 --tag gemma --initial $E/jobs/pilot_gemma.jsonl 2>&1 | tee -a $R/pod.log
waitfor /root/dl_qwen.ok
$PY --model Qwen/Qwen3.6-27B --revision $QREV --fmt qwen --tag qwen --initial $E/jobs/pilot_qwen.jsonl --inbox $IN/qwen --wait-minutes 25 2>&1 | tee -a $R/pod.log
$PY --model google/gemma-4-31B-it --revision $GREV --fmt gemma4 --tag gemma --inbox $IN/gemma --wait-minutes 10 2>&1 | tee -a $R/pod.log
tail -3 $R/download.log >> $R/pod.log
echo "[s2] pod done" | tee -a $R/pod.log
