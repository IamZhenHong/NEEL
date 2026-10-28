#!/usr/bin/env bash
# CC-F-A1 training + serving pod (deviation 8: parallel pods). One A100 80GB. Launch with launch_trainpod.sh.
#   1. pod_serve.sh's set-up (vLLM venv, Qwen3-14B @40c06982) with generation skipped (GEN_DONE) and /root/PAUSE set
#   2. in the background, once /root/base_path exists: pod_train_seq2.sh trains LIST (VERSION:SEED ...) in turn on
#      GPU 0 with STEPS = 1,000 (pod S's timing smoke, the same GPU type: 2.593 s/step), DATA / DATA_SHA as given;
#      each adapter goes to /root/adapters_in/<name> and /root/loras.txt; at the end /root/PAUSE is removed
#   3. pod_serve.sh's server: base "qwen3-14b" + /root/loras.txt, the same flags as pod S, until /root/STOP or the
#      deadline; /root/PAUSE stops it again (e.g. to train V5 on this pod)
# env: T0, MAXMIN, LIST, DATA_SHA (required); DATA (default data/train_versions.jsonl.gz)
set -uo pipefail
cd "$(dirname "$0")/../../.."
H=character_conduct/final_bets/a1_train
: "${T0:?}" "${MAXMIN:?}" "${LIST:?}" "${DATA_SHA:?}"
OUT=$H/results/serve; OUTT=$H/results/train
mkdir -p "$OUT/gen" "$OUTT"; touch "$OUT/gen/GEN_DONE" /root/PAUSE
echo '{"steps": 1000, "rule": "pod S timing smoke (A100 SXM, 2.593 s/step), deviation 8"}' > "$OUTT/steps.json"
( for i in $(seq 1 240); do [ -f /root/base_path ] && break; sleep 15; done
  echo "[a1] training $LIST from $(date -u +%H:%M)"
  DATA=${DATA:-$H/data/train_versions.jsonl.gz} bash "$H/pod_train_seq2.sh" "$OUTT" "$DATA_SHA" $LIST
  echo "[a1] training block exit $? $(date -u +%H:%M)" ) > /root/train.log 2>&1 &
T0=$T0 MAXMIN=$MAXMIN OUT=$OUT bash "$H/pod_serve.sh"
